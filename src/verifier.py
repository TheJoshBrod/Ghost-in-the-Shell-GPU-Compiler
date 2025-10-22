"""
src/verifier.py
Validates a generated CUDA kernel by compiling it as a PyTorch C++ extension
and comparing its tensor output against a ground-truth tensor.
"""

import torch
import tempfile
import os
import shutil
from torch.utils.cpp_extension import load
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
def validate_kernel(
    generated_cu_code: str,
    generated_cpp_code: str,
    input_tensor_path: str,
    ground_truth_path: str,
    timeout_seconds: int = 120
) -> tuple[bool, bool, str]:
    """
    Validates a CUDA kernel using the PyTorch C++ extension API.
    ...
    """

    tmpdir = tempfile.mkdtemp(prefix="gins_verifier_")
    log_message = ""

    # --- FIX 1: Define all status variables at the top ---
    call_success = False
    exec_success = False
    runtime_success = False # Add this for clarity

    try:
        cu_path = os.path.join(tmpdir, "kernel.cu")
        
        # --- FIX 2: Save the wrapper as a .cu file, not .cpp ---
        # This tells torch.utils.load() to use NVCC (the CUDA compiler)
        wrapper_cu_path = os.path.join(tmpdir, "wrapper.cu") 

        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(generated_cu_code)
        
        with open(wrapper_cu_path, "w", encoding="utf-8") as f:
            f.write(generated_cpp_code)

        # --- 2. Stage 1: Call Status (Compilation) ---
        log.info(f"Attempting JIT compilation in {tmpdir}...")
        try:
            module = load(
                name=f"generated_module_{os.path.basename(tmpdir)}",
                
                # --- FIX 3: Pass BOTH .cu files to the loader ---
                sources=[wrapper_cu_path, cu_path],
                
                build_directory=tmpdir,
                verbose=True, 
            )
            call_success = True # <-- Compilation succeeded
            log_message = "Compilation successful.\n"
            log.info("Compilation successful.")

        except Exception as e:
            # --- FIX 4: Correctly handle compilation failure ---
            call_success = False
            exec_success = False
            log_message = f"Compilation Failed (Call Status=False):\n{e}"
            log.warning(log_message)
            # Return here; no need to run exec status
            return call_success, exec_success, log_message
        
        # --- 3. Stage 2: Execution Status (Correctness) ---
        # This code only runs if call_success == True
        log.info("Running execution status check...")
        try:
            # Load ground truth and inputs
            inputs = torch.load(input_tensor_path)
            ground_truth = torch.load(ground_truth_path).cuda()
            
            # Ensure inputs are on GPU
            cuda_inputs = [t.cuda() for t in inputs]
            
            # Create an empty output tensor with the correct shape/type
            output_generated = torch.empty_like(ground_truth)
            
            # Call the compiled kernel function
            module.launch(*cuda_inputs, output_generated)
            
            # Kernel executed without a runtime error
            runtime_success = True

        except Exception as e:
            # --- FIX 5: Handle runtime (not compilation) errors ---
            runtime_success = False
            exec_success = False
            log_message += f"Kernel Runtime Error (Exec Status=False):\n{e}"
            log.warning(log_message)
            # 'call_success' is still True, so we return it
            return call_success, exec_success, log_message

        # --- 4. Final Comparison ---
        if runtime_success:
            try:
                # Use numerical tolerance checking
                is_correct = torch.allclose(output_generated, ground_truth, atol=1e-2, rtol=1e-1)
                
                if is_correct:
                    exec_success = True
                    log_message += "Validation Successful (Exec Status=True): Outputs match."
                    log.info("Validation Successful.")
                else:
                    exec_success = False
                    diff = torch.abs(output_generated - ground_truth)
                    log_message += (
                        f"Correctness Mismatch (Exec Status=False):\n"
                        f"Max difference: {diff.max().item()}\n"
                        f"Mean difference: {diff.mean().item()}"
                    )
                    log.warning(log_message)

            except Exception as e:
                exec_success = False
                log_message += f"Output Comparison Error (Exec Status=False):\n{e}"
                log.warning(log_message)

        # Final return
        return call_success, exec_success, log_message

    finally:
        # Clean up the temporary directory
        if os.path.exists(tmpdir):
            try:
                shutil.rmtree(tmpdir)
                log.info(f"Cleaned up {tmpdir}")
            except Exception as e:
                log.error(f"Failed to clean up {tmpdir}: {e}")