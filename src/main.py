"""
Walks through each kernel in the benchmark to:
- Monitor kernel and aten calls
- Generate an initial output IR 
- Initiate a Validation loop until correctly generated kernel  
"""

import sys
import json

import os
os.makedirs("generated_kernels", exist_ok=True)

import src.monitor
import src.generator
import src.verifier

MAX_ATTEMPTS = 5

def generate_kernel(program: dict[str], counter: int):
    """Monitors operations, generates corresponding kernel, and validates correctness for each sample of benchmark.

    Args:
        inputs (list[str]): List of inputs that the sample takes (values, size, etc.)
        code (str): Literal PyTorch code to be executed
    """

    output, aten_output, kernel_output = src.monitor.extract_op_details(program)
    
    op_details = f"aten output:\n{aten_output}\n\n\nkernel output:\n{kernel_output}"

    generated_code = src.generator.gemini_generator(op_details)
    
    for i in range(MAX_ATTEMPTS):
        is_valid, feedback = src.verifier.verify_cuda(generated_code, str(output))
        
        with open(f"generated_kernels/{os.path.basename(sys.argv[1])}_{counter}_iter{i}.txt", "w") as f:
            f.write(f"feedback {i}: {feedback}\ncode {i}: {generated_code}\n\n\n")
            
        if is_valid:
            with open(f"generated_kernels/{sys.argv[1]}_{counter}.cu", "w") as f:
                f.write(generated_code)
            return

        generated_code = src.generator.gemini_fixer(generated_code, feedback, op_details)

    with open(f"generated_kernels/{sys.argv[1]}_{counter}.cu", "w") as f:
        f.write(generated_code)

def main():
    """Loops through each sample in benchmark."""
    
    benchmarks = []
    with open(sys.argv[1], "r") as f:
        benchmarks = json.load(f)

    counter = 0
    for program in benchmarks:
        generate_kernel(program, counter)
        counter += 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing benchmark file `python3 src/main.py <benchmark file>")
        sys.exit(1)
    
    
    main()