"""
Main pipeline for CUDA kernel generation and validation.

Walks through each operation in a benchmark to:
1. Monitor kernel and ATen calls
2. Generate CUDA kernel code
3. Validate correctness through iterative refinement
"""

import sys
import json
import torch
from pathlib import Path

import src.monitor
import src.generator
import src.verifier

# Configuration
MAX_ATTEMPTS = 5
OUTPUT_DIR = Path("generated_kernels")


def setup_output_directory():
    """Create output directory for logs and generated kernels."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def initialize_context(definitions: list[dict]) -> dict:
    """
    Initialize execution context with variable definitions.
    
    Args:
        definitions (list[dict]): List of dicts with 'variable' and 'value' keys
        
    Returns:
        dict: Execution context with torch and defined variables
    """
    context = {"torch": torch}
    for definition in definitions:
        exec(f"{definition['variable']} = {definition['value']}", context)
    return context


def save_tensors(input_tensors: list[torch.Tensor], 
                 ground_truth: torch.Tensor, 
                 op_id: str) -> tuple[str, str]:
    """
    Save input and ground truth tensors to disk.
    
    Returns:
        Tuple of (input_path, ground_truth_path)
    """
    input_path = OUTPUT_DIR / f"{op_id}_inputs.pt"
    gold_path = OUTPUT_DIR / f"{op_id}_gold.pt"
    
    torch.save(input_tensors, input_path)
    torch.save(ground_truth, gold_path)
    
    return str(input_path), str(gold_path)


def log_attempt(op_id: str, iteration: int, cu_code: str, feedback: str):
    """Log kernel code and feedback for a validation attempt."""
    log_path = OUTPUT_DIR / f"{op_id}_iter{iteration}.log"
    with open(log_path, "w") as f:
        f.write(f"--- FEEDBACK ---\n{feedback}\n\n")
        f.write(f"--- KERNEL.CU ---\n{cu_code}\n\n")


def save_final_kernel(op_id: str, cu_code: str, success: bool):
    """Save the final kernel code (successful or failed)."""
    suffix = "_kernel_final.cu" if success else "_kernel_final_FAILED.cu"
    final_path = OUTPUT_DIR / f"{op_id}{suffix}"
    with open(final_path, "w") as f:
        f.write(cu_code)


def validate_with_retries(cu_code: str, 
                         input_path: str, 
                         gold_path: str,
                         op_details: dict,
                         op_id: str,
                         op_index: int) -> tuple[bool, str]:
    """
    Attempt to validate and fix kernel code up to MAX_ATTEMPTS times.
    
    Returns:
        Tuple of (is_valid, final_cu_code)
    """
    current_code = cu_code
    
    for attempt in range(MAX_ATTEMPTS):
        print(f"--- Attempt {attempt + 1}/{MAX_ATTEMPTS} for Op {op_index} ---")
        
        # Validate current kernel
        call_success, exec_success, feedback = src.verifier.validate_kernel(
            current_code, input_path, gold_path
        )
        
        # Log the attempt
        log_attempt(op_id, attempt, current_code, feedback)
        
        is_valid = call_success and exec_success
        
        if is_valid:
            print(f"✓ Validation SUCCESSFUL for Op {op_index} on attempt {attempt + 1}!")
            return True, current_code
        
        # Try to fix if not the last attempt
        if attempt < MAX_ATTEMPTS - 1:
            print("✗ Validation FAILED. Attempting to fix...")
            try:
                current_code = src.generator.gemini_fixer(
                    cu_code=current_code,
                    error=feedback,
                    msg=op_details   
                )
            except Exception as e:
                import traceback
                print(f"Fixer failed: {e}. Stopping attempts for this op.")
                traceback.print_exc()
                print("Stopping attempts for this op.")
                break
    
    print(f"✗ Failed to generate correct kernel for Op {op_index} after {MAX_ATTEMPTS} attempts.")
    return False, current_code


def process_operation(op: dict, 
                     op_index: int, 
                     context: dict, 
                     benchmark_id: str) -> bool:
    """
    Process a single operation: profile, generate, validate.
    
    Returns:
        True if operation was successfully processed and executed
    """
    # Parse operation structure
    try:
        assignment_name = op["assignment"]
        op_name = op["operation"]
        input_names = op["inputs"]
    except KeyError:
        print(f"⚠ Skipping operation {op_index}: Missing required keys.")
        return False
    
    # Build execution string
    op_call_str = f"{op_name}({', '.join(input_names)})"
    full_exec_string = f"{assignment_name} = {op_call_str}"
    
    print(f"\n{'='*60}")
    print(f"Processing Op {op_index + 1}: {full_exec_string}")
    print(f"{'='*60}")
    
    # Get input tensors from context
    try:
        input_tensors = [context[name] for name in input_names]
    except KeyError as e:
        print(f"✗ Error: Variable {e} not found in context. Skipping op.")
        return False
    
    # Profile operation
    try:
        ground_truth, op_details = src.monitor.profile_single_op(
            context, full_exec_string
        )
    except Exception as e:
        print(f"✗ Failed to profile op: {e}")
        return False
    
    # Save tensors for verification
    op_id = f"{benchmark_id}_op{op_index}"
    input_path, gold_path = save_tensors(input_tensors, ground_truth, op_id)
    
    # Generate initial kernel
    print("Generating initial kernel code...")
    try:
        cu_code = src.generator.gemini_generator(op_details)
    except Exception as e:
        print(f"✗ Initial generation failed: {e}")
        return False
    
    # Validation loop with retries
    is_valid, final_code = validate_with_retries(
        cu_code, input_path, gold_path, op_details, op_id, op_index
    )
    
    # Save final kernel
    save_final_kernel(op_id, final_code, is_valid)
    
    # Execute operation to update context for next op
    print(f"Executing '{full_exec_string}' to update context.")
    exec(full_exec_string, context)
    
    return True


def process_benchmark(program: dict, benchmark_counter: int, benchmark_name: str):
    """
    Process all operations in a benchmark program.
    
    Args:
        program: Dict containing 'definitions' and 'operations'
        benchmark_counter: Index of this benchmark
        benchmark_name: Name of the benchmark file
    """
    benchmark_id = f"{benchmark_name}_{benchmark_counter}"
    
    # Initialize execution context
    context = initialize_context(program["definitions"])
    
    # Process each operation
    operations = program["operations"]
    for op_index, op in enumerate(operations):
        process_operation(op, op_index, context, benchmark_id)


def load_benchmarks(benchmark_path: str) -> tuple[list[dict], str]:
    """
    Load benchmarks from JSON file.
    
    Returns:
        Tuple of (benchmarks_list, benchmark_name)
    """
    with open(benchmark_path, "r") as f:
        benchmarks = json.load(f)
    
    benchmark_name = Path(benchmark_path).stem
    return benchmarks, benchmark_name


def main():
    """Main entry point: load benchmarks and process each one."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <benchmark_file.json>")
        sys.exit(1)
    
    setup_output_directory()
    
    benchmark_file = sys.argv[1]
    benchmarks, benchmark_name = load_benchmarks(benchmark_file)
    
    print(f"\n{'='*60}")
    print(f"Processing benchmark: {benchmark_name}")
    print(f"Total programs: {len(benchmarks)}")
    print(f"{'='*60}\n")
    
    for i, program in enumerate(benchmarks):
        print(f"\n{'#'*60}")
        print(f"# Benchmark {i + 1}/{len(benchmarks)}")
        print(f"{'#'*60}")
        process_benchmark(program, i, benchmark_name)
    
    print(f"\n{'='*60}")
    print("✓ All benchmarks processed!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()