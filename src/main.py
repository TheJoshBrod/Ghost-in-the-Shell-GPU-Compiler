"""
Walks through each kernel in the benchmark to:
- Monitor kernel and aten calls
- Generate an initial output IR 
- Initiate a Validation loop until correctly generated kernel  
"""

import src.monitor
import src.generator
import src.verifier

MAX_ATTEMPTS = 5

def generate_kernel(inputs: list[str], code: str):
    """Monitors operations, generates corresponding kernel, and validates correctness for each sample of benchmark.

    Args:
        inputs (list[str]): List of inputs that the sample takes (values, size, etc.)
        code (str): Literal PyTorch code to be executed
    """
 
    output, aten_output, kernel_output = src.monitor.extract_op_details(inputs, code)
    
    op_details = f"aten output:\n{aten_output}\n\n\nkernel output:\n{kernel_output}"
    generated_code = src.generator.ollama_generator(op_details)
    
    for i in range(MAX_ATTEMPTS):
        is_valid, feedback = src.verifier.verify_cuda(generated_code, output)

        if is_valid:
            with open("final_kernel.py", "w") as f:
                f.write(generated_code)
            return

        generated_code = src.generator.ollama_fixer(generated_code, feedback, op_details)



def main():
    """Loops through each sample in benchmark."""
    
    benchmark = [] # TODO
    for inputs, code in benchmark:
        generate_kernel(inputs, code)


if __name__ == "__main__":
    main()