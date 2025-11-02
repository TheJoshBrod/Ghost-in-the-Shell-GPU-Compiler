"""
src/verifier.py
Validates a generated CUDA kernel by compiling it as a PyTorch C++ extension
and comparing its tensor output against a ground-truth tensor.
"""

import torch
import tempfile
import os
import shutil
import time
from torch.utils.cpp_extension import load
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def validate_kernel(
    generated_cu_code: str,
    input_tensor_path: str,
    ground_truth_path: str,
    timeout_seconds: int = 120
) -> tuple[bool, bool, str, dict]:
    """
    Validates a single-file CUDA kernel using the PyTorch C++ extension API.
    
    Returns:
        tuple: (call_success, exec_success, log_message, metrics)
            metrics dict contains: {
                'execution_time_ms': float,
                'memory_allocated_mb': float,
                'memory_reserved_mb': float,
                'peak_memory_mb': float
            }
    """

    tmpdir = tempfile.mkdtemp(prefix="gins_verifier_")
    log_message = ""

    call_success = False
    exec_success = False
    runtime_success = False
    
    # Initialize metrics dictionary
    metrics = {
        'execution_time_ms': None,
        'memory_allocated_mb': None,
        'memory_reserved_mb': None,
        'peak_memory_mb': None
    }

    try:
        # --- 1. Stage 1: Write Code to File ---
        cu_path = os.path.join(tmpdir, "kernel.cu")
        
        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(generated_cu_code)

        # --- 2. Stage 1: Call Status (Compilation) ---
        log.info(f"Attempting JIT compilation in {tmpdir}...")
        try:
            module = load(
                name=f"generated_module_{os.path.basename(tmpdir)}",
                sources=[cu_path],
                build_directory=tmpdir,
                verbose=True, 
            )
            call_success = True
            log_message = "Compilation successful.\n"
            log.info("Compilation successful.")

        except Exception as e:
            # --- Handle compilation failure ---
            call_success = False
            exec_success = False
            log_message = f"Compilation Failed (Call Status=False):\n{e}"
            log.warning(log_message)
            return call_success, exec_success, log_message, metrics
        
        # --- 3. Stage 2: Execution Status (Correctness) ---
        log.info("Running execution status check...")
        try:
            # Load ground truth and inputs
            inputs = torch.load(input_tensor_path)
            ground_truth = torch.load(ground_truth_path).cuda()
            
            # Prepare inputs: move tensors to CUDA, keep scalars as-is
            if isinstance(inputs, (list, tuple)):
                cuda_inputs = []
                for item in inputs:
                    if torch.is_tensor(item):
                        cuda_inputs.append(item.cuda())
                    elif isinstance(item, (int, float, bool, str)):
                        # Keep scalar types as-is
                        cuda_inputs.append(item)
                        log.info(f"Passing non-tensor argument: {type(item).__name__} = {item}")
                    else:
                        log.warning(f"Skipping unsupported input type: {type(item)}")
            elif torch.is_tensor(inputs):
                cuda_inputs = [inputs.cuda()]
            else:
                # Single scalar input
                if isinstance(inputs, (int, float, bool, str)):
                    cuda_inputs = [inputs]
                else:
                    exec_success = False
                    log_message += "No valid inputs found (Exec Status=False)"
                    log.warning(log_message)
                    return call_success, exec_success, log_message, metrics
            
            # Reset peak memory stats before kernel execution
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            # Record memory before execution
            mem_before = torch.cuda.memory_allocated()
            
            # Time the kernel execution
            start_time = time.perf_counter()
            
            # Call the compiled kernel
            output_generated = module.launch(*cuda_inputs)
            
            # Ensure all CUDA operations complete
            torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Record memory after execution
            mem_after = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            reserved_mem = torch.cuda.memory_reserved()
            
            # Calculate metrics
            execution_time_ms = (end_time - start_time) * 1000
            memory_allocated_mb = (mem_after - mem_before) / (1024 ** 2)
            memory_reserved_mb = reserved_mem / (1024 ** 2)
            peak_memory_mb = peak_mem / (1024 ** 2)
            
            # Store metrics
            metrics['execution_time_ms'] = execution_time_ms
            metrics['memory_allocated_mb'] = memory_allocated_mb
            metrics['memory_reserved_mb'] = memory_reserved_mb
            metrics['peak_memory_mb'] = peak_memory_mb
            
            log.info(f"Execution time: {execution_time_ms:.3f} ms")
            log.info(f"Memory allocated: {memory_allocated_mb:.3f} MB")
            log.info(f"Peak memory: {peak_memory_mb:.3f} MB")
            
            # Move to same device as ground truth if needed
            if not output_generated.is_cuda:
                output_generated = output_generated.cuda()
            
            # Kernel executed without a runtime error
            runtime_success = True

        except Exception as e:
            # --- Handle runtime errors ---
            runtime_success = False
            exec_success = False
            log_message += f"Kernel Runtime Error (Exec Status=False):\n{e}"
            log.warning(log_message)
            return call_success, exec_success, log_message, metrics

        # --- 4. Final Comparison ---
        if runtime_success:
            try:
                # Use numerical tolerance checking
                is_correct = torch.allclose(output_generated, ground_truth, atol=1e-2, rtol=1e-1)
                
                if is_correct:
                    exec_success = True
                    log_message += (
                        f"Validation Successful (Exec Status=True): Outputs match.\n"
                        f"Execution time: {metrics['execution_time_ms']:.3f} ms\n"
                        f"Memory allocated: {metrics['memory_allocated_mb']:.3f} MB\n"
                        f"Peak memory: {metrics['peak_memory_mb']:.3f} MB"
                    )
                    log.info("Validation Successful.")
                else:
                    exec_success = False
                    diff = torch.abs(output_generated - ground_truth)
                    log_message += (
                        f"Correctness Mismatch (Exec Status=False):\n"
                        f"Max difference: {diff.max().item()}\n"
                        f"Mean difference: {diff.mean().item()}\n"
                        f"Execution time: {metrics['execution_time_ms']:.3f} ms\n"
                        f"Memory allocated: {metrics['memory_allocated_mb']:.3f} MB"
                    )
                    log.warning(log_message)

            except Exception as e:
                exec_success = False
                log_message += f"Output Comparison Error (Exec Status=False):\n{e}"
                log.warning(log_message)

        # Final return
        return call_success, exec_success, log_message, metrics

    finally:
        # Clean up the temporary directory
        if os.path.exists(tmpdir):
            try:
                shutil.rmtree(tmpdir)
                log.info(f"Cleaned up {tmpdir}")
            except Exception as e:
                log.error(f"Failed to clean up {tmpdir}: {e}")