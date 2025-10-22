"""
Walks through each kernel in the benchmark to:
- Monitor kernel and aten calls
- Generate an initial output IR 
- Initiate a Validation loop until correctly generated kernel  
"""

import os
import sys
import json
import torch

# Create output directory for logging and valid kernels
os.makedirs("generated_kernels", exist_ok=True)

import src.monitor
import src.generator
import src.verifier

MAX_ATTEMPTS = 5

def process_benchmark(program: dict, benchmark_counter: int, benchmark_name: str):
    """Monitors operations, generates corresponding kernel, and validates correctness for each sample of benchmark.

    Args:
        inputs (list[str]): List of inputs that the sample takes (values, size, etc.)
        code (str): Literal PyTorch code to be executed
    """

    # Generates the initial input tensor
    context = {"torch": torch}
    for definition in program["definitions"]:
        exec(f"{definition['variable']} = {definition['value']}", context)

    for op_index, op in enumerate(program["operations"]):
        
        # --- [THIS IS THE NEW LOGIC] ---
        # Assumes new JSON format
        try:
            assignment_name = op["assignment"]
            op_name = op["operation"]
            input_names = op["inputs"]
        except KeyError:
            print(f"Skipping operation {op_index}: Benchmark not in new JSON format (missing 'assignment', 'operation', or 'inputs').")
            continue
            
        # Build the operation string
        op_call_str = f"{op_name}({', '.join(input_names)})"
        full_exec_string = f"{assignment_name} = {op_call_str}"
        
        print(f"\n--- Processing Op {op_index+1}/{len(program['operations'])}: {full_exec_string} ---")
        
        # 3. Get the *specific* inputs for THIS operation
        try:
            input_tensors = [context[name] for name in input_names]
        except KeyError as e:
            print(f"Error: Variable {e} not found in context. Skipping op.")
            continue
            
        # 4. Profile *only* this single operation
        try:
            ground_truth_tensor, op_details = src.monitor.profile_single_op(
                context, full_exec_string
            )
        except Exception as e:
            print(f"Failed to profile op {op_index}: {e}. Skipping.")
            continue
            
        # --- [Your existing validation loop goes here, with minor changes] ---
        
        # 5. Save tensors for the verifier
        op_id_str = f"{benchmark_name}_{benchmark_counter}_op{op_index}"
        input_path = f"generated_kernels/{op_id_str}_inputs.pt"
        gold_path = f"generated_kernels/{op_id_str}_gold.pt"

        torch.save(input_tensors, input_path)
        torch.save(ground_truth_tensor, gold_path)
        
        # 6. Initial Generation
        print("Generating initial code...")
        try:
            cu_code, cpp_code = src.generator.gemini_generator(op_details)
        except Exception as e:
            print(f"Initial generation failed: {e}. Skipping op.")
            continue

        # 7. Validation Loop
        is_valid = False
        for i in range(MAX_ATTEMPTS):
            print(f"--- Attempt {i+1}/{MAX_ATTEMPTS} for Op {op_index} ---")

            call_success, exec_success, feedback = src.verifier.validate_kernel(
                cu_code, cpp_code, input_path, gold_path
            )
            
            # Log the attempt
            with open(f"generated_kernels/{op_id_str}_iter{i}.log", "w") as f:
                f.write(f"--- FEEDBACK ---\n{feedback}\n\n")
                f.write(f"--- KERNEL.CU ---\n{cu_code}\n\n")
                f.write(f"--- WRAPPER.CPP ---\n{cpp_code}\n\n")
                
            is_valid = call_success and exec_success
            
            if is_valid:
                print(f"Validation SUCCESSFUL for Op {op_index} on attempt {i+1}!")
                with open(f"generated_kernels/{op_id_str}_final.cu", "w") as f:
                    f.write(cu_code)
                with open(f"generated_kernels/{op_id_str}_final.cpp", "w") as f:
                    f.write(cpp_code)
                break # Success! Move to next operation
            
            if i < MAX_ATTEMPTS - 1:
                print("Validation FAILED. Attempting to fix...")
                try:
                    cu_code, cpp_code = src.generator.gemini_fixer(
                        cu_code, cpp_code, feedback, op_details
                    )
                except Exception as e:
                    print(f"Fixer failed: {e}. Stopping attempts for this op.")
                    break
        
        if not is_valid:
            print(f"Failed to generate correct kernel for Op {op_index} after {MAX_ATTEMPTS} attempts.")
            # Save failed attempt
            with open(f"generated_kernels/{op_id_str}_final_FAILED.cu", "w") as f:
                f.write(cu_code)
            with open(f"generated_kernels/{op_id_str}_final_FAILED.cpp", "w") as f:
                f.write(cpp_code)

        # 8. CRITICAL: Execute the op in the main context
        # This makes its output (e.g., 'c') available for the next op (e.g., 'd = sin(c)')
        print(f"Executing '{full_exec_string}' to update main context.")
        exec(full_exec_string, context)

def main():
    """Loops through each sample in benchmark."""
    
    benchmark_file = sys.argv[1]
    benchmark_name = os.path.basename(benchmark_file).split('.')[0]

    benchmarks = []
    with open(sys.argv[1], "r") as f:
        benchmarks = json.load(f)

    for i, program in enumerate(benchmarks):
        process_benchmark(program, i, benchmark_name)
        counter += 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing benchmark file `python3 src/main.py <benchmark file>")
        sys.exit(1)
    main()
