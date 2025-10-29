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
import csv
from pathlib import Path
from statistics import mean, median

import src.monitor
import src.generator
import src.verifier

# Configuration
MAX_ATTEMPTS = 5
OUTPUT_BASE_DIR = Path("generated_kernels")

# Global statistics tracking
benchmark_stats = []


def setup_operation_directory(benchmark_name: str, op_name: str) -> Path:
    """
    Create nested output directory for a specific operation.
    
    Args:
        benchmark_name: Name of the benchmark
        op_name: Name of the operation/test
        
    Returns:
        Path to the operation directory
    """
    op_dir = OUTPUT_BASE_DIR / benchmark_name / op_name
    op_dir.mkdir(parents=True, exist_ok=True)
    return op_dir


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
                 op_dir: Path) -> tuple[str, str]:
    """
    Save input and ground truth tensors to disk.
    
    Returns:
        Tuple of (input_path, ground_truth_path)
    """
    input_path = op_dir / "inputs.pt"
    gold_path = op_dir / "gold.pt"
    
    torch.save(input_tensors, input_path)
    torch.save(ground_truth, gold_path)
    
    return str(input_path), str(gold_path)


def log_attempt(op_dir: Path, iteration: int, cu_code: str, feedback: str):
    """Log kernel code and feedback for a validation attempt."""
    feedback_path = op_dir / f"feedback_{iteration}.md"
    with open(feedback_path, "w") as f:
        f.write(f"# Feedback - Iteration {iteration}\n\n")
        f.write(feedback)


def save_kernel_iteration(op_dir: Path, iteration: int, cu_code: str):
    """Save kernel code for a specific iteration."""
    kernel_path = op_dir / f"kernel_iter{iteration}.cu"
    with open(kernel_path, "w") as f:
        f.write(cu_code)


def save_final_kernel(op_dir: Path, cu_code: str, success: bool):
    """Save the final kernel code."""
    kernel_path = op_dir / "kernel.cu"
    with open(kernel_path, "w") as f:
        f.write(cu_code)
    
    # Also save a status file
    status_path = op_dir / "status.txt"
    with open(status_path, "w") as f:
        f.write("SUCCESS" if success else "FAILED")


def validate_with_retries(cu_code: str, 
                         input_path: str, 
                         gold_path: str,
                         op_details: dict,
                         op_dir: Path,
                         op_index: int) -> tuple[bool, str, int]:
    """
    Attempt to validate and fix kernel code up to MAX_ATTEMPTS times.
    
    Returns:
        Tuple of (is_valid, final_cu_code, attempts_until_success)
        attempts_until_success is -1 if failed
    """
    current_code = cu_code
    
    for attempt in range(MAX_ATTEMPTS):
        print(f"--- Attempt {attempt + 1}/{MAX_ATTEMPTS} for Op {op_index} ---")
        
        # Validate current kernel
        call_success, exec_success, feedback = src.verifier.validate_kernel(
            current_code, input_path, gold_path
        )
        
        # Log the attempt
        log_attempt(op_dir, attempt, current_code, feedback)
        save_kernel_iteration(op_dir, attempt, current_code)
        
        is_valid = call_success and exec_success
        
        if is_valid:
            print(f"✓ Validation SUCCESSFUL for Op {op_index} on attempt {attempt + 1}!")
            return True, current_code, attempt + 1
        
        # Try to fix if not the last attempt
        if attempt < MAX_ATTEMPTS - 1:
            print("✗ Validation FAILED. Attempting to fix...")
            try:
                current_code = src.generator.chatgpt_fixer(
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
    return False, current_code, -1


def process_operation(op: dict, 
                     op_index: int, 
                     context: dict, 
                     benchmark_name: str,
                     test_name: str) -> bool:
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
    
    # Use test name as operation directory name
    op_dir_name = test_name
    
    print(f"\n{'='*60}")
    print(f"Processing Op {op_index + 1}: {full_exec_string}")
    print(f"{'='*60}")
    
    # Setup directory for this operation
    op_dir = setup_operation_directory(benchmark_name, op_dir_name)
    print(f"Output directory: {op_dir}")
    
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
    input_path, gold_path = save_tensors(input_tensors, ground_truth, op_dir)
    
    # Generate initial kernel
    print("Generating initial kernel code...")
    try:
        cu_code = src.generator.chatgpt_generator(op_details)
    except Exception as e:
        print(f"✗ Initial generation failed: {e}")
        return False
    
    # Validation loop with retries
    is_valid, final_code, attempts = validate_with_retries(
        cu_code, input_path, gold_path, op_details, op_dir, op_index
    )
    
    # Save final kernel
    save_final_kernel(op_dir, final_code, is_valid)
    
    # Record statistics
    global benchmark_stats
    benchmark_stats.append({
        'test_name': test_name,
        'operation': op_name,
        'success': is_valid,
        'attempts': attempts
    })

    # Execute operation to update context for next op
    print(f"Executing '{full_exec_string}' to update context.")
    exec(full_exec_string, context)
    
    return True


def process_benchmark(program: dict, benchmark_counter: int, benchmark_name: str):
    """
    Process all operations in a benchmark program.
    
    Args:
        program: Dict containing 'name', 'definitions' and 'operations'
        benchmark_counter: Index of this benchmark
        benchmark_name: Name of the benchmark file
    """
    # Get test name from program
    test_name = program.get("name", f"test{benchmark_counter}")
    
    # Initialize execution context
    context = initialize_context(program["definitions"])
    
    # Process each operation
    operations = program["operations"]
    for op_index, op in enumerate(operations):
        process_operation(op, op_index, context, benchmark_name, test_name)


def save_statistics_csv(benchmark_name: str):
    """
    Save benchmark statistics to a CSV file.
    Includes median and mean attempts for successful runs, and total success rate.
    """
    if not benchmark_stats:
        print("No statistics to save.")
        return
    
    # Calculate metrics
    successful_runs = [s for s in benchmark_stats if s['success']]
    total_runs = len(benchmark_stats)
    success_count = len(successful_runs)
    success_rate = (success_count / total_runs * 100) if total_runs > 0 else 0
    
    # Calculate median and mean attempts for successful runs only
    if successful_runs:
        attempts_list = [s['attempts'] for s in successful_runs]
        median_attempts = median(attempts_list)
        mean_attempts = mean(attempts_list)
    else:
        median_attempts = 0
        mean_attempts = 0
    
    # Save summary CSV
    summary_path = OUTPUT_BASE_DIR / benchmark_name / "summary_statistics.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Operations', total_runs])
        writer.writerow(['Successful Operations', success_count])
        writer.writerow(['Success Rate (%)', f'{success_rate:.2f}'])
        writer.writerow(['Median Attempts (Success Only)', f'{median_attempts:.2f}'])
        writer.writerow(['Mean Attempts (Success Only)', f'{mean_attempts:.2f}'])
    
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"  Total Operations: {total_runs}")
    print(f"  Successful: {success_count}")
    print(f"  Success Rate: {success_rate:.2f}%")
    print(f"  Median Attempts (successful): {median_attempts:.2f}")
    print(f"  Mean Attempts (successful): {mean_attempts:.2f}")
    print(f"  Saved to: {summary_path}")
    print(f"{'='*60}\n")
    
    # Save detailed CSV with per-operation stats
    detail_path = OUTPUT_BASE_DIR / benchmark_name / "detailed_statistics.csv"
    with open(detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test Name', 'Operation', 'Success', 'Attempts'])
        for stat in benchmark_stats:
            attempts_str = str(stat['attempts']) if stat['attempts'] != -1 else 'Failed'
            writer.writerow([
                stat['test_name'],
                stat['operation'],
                'Yes' if stat['success'] else 'No',
                attempts_str
            ])
    
    print(f"Detailed statistics saved to: {detail_path}\n")


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
    print(f"Results saved to: {OUTPUT_BASE_DIR}")
    print(f"{'='*60}\n")
    
    # Save statistics CSV
    save_statistics_csv(benchmark_name)


if __name__ == "__main__":
    main()