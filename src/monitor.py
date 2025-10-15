"""Monitors and Preprocesses PyTorch Aten and CUDA Kernel Abstraction Layer Calls for Generator"""
import torch
import time

if not torch.cuda.is_available():
    raise RuntimeError("This example requires a CUDA-enabled PyTorch installation.")

def handle_trace(prof):
    """
    This callback provides a two-level summary of the profiler trace:
    1. A high-level summary of PyTorch 'aten' operators.
    2. A detailed breakdown of the specific CUDA kernels that were launched.
    """
    print("Hello world")
    aggregated_events = prof.key_averages(group_by_input_shape=True)

    # --- 1. High-Level Operator Summary ---
    for event in aggregated_events:
        # Filter for the main PyTorch operators
        if event.key.startswith("aten::") and event.self_device_time_total > 0:
            print(f"[Op: {event.key}]")
            
            if event.input_shapes:
                print(f"  Inputs: {event.input_shapes}")
            else:
                print(f"  Inputs: (Not available in this trace)")

            print(f"  Total Device Time (ms): {event.self_device_time_total / 1000.0:.3f}")

    # --- 2. Detailed CUDA Kernel Breakdown ---
    for event in aggregated_events:
        # Filter for the low-level kernel names. These do not start with 'aten::'
        # and are not internal profiler steps.
        if not event.key.startswith("aten::") and "ProfilerStep" not in event.key and event.self_device_time_total > 0:
            print(f"[Kernel: {event.key}]")
            print(f"  Device Time (ms): {event.self_device_time_total / 1000.0:.3f}")


def extract_op_details(inputs: list[str], code: str):
    """Extracts relevant information of sample PyTorch code including: 
    - High level kernel PyTorch operators
    - Names of PyTorch’s internal CUDA kernel/cuBLAS kernels/etc.
    - Correct output given the input
    
    Args:
        inputs (list[str]): List of inputs that the sample takes (values, size, etc.)
        code (str): Literal PyTorch code to be executed
    
    Output:
        str: Correct value of the PyTorch sample
        str: Formatted string of high and low level representation of sample
    """

    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")

    print("\n[Main Thread] Starting profiled execution...")

    # --- Define schedule parameters explicitly ---
    wait_steps = 1
    warmup_steps = 1
    active_steps = 2
    repeat_cycles = 1
    # -------------------------------------------

    # Create the schedule function
    schedule = torch.profiler.schedule(
        wait=wait_steps, 
        warmup=warmup_steps, 
        active=active_steps, 
        repeat=repeat_cycles
    )

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=handle_trace,
        record_shapes=True
    ) as p:
        # --- Calculate the total number of steps for the loop ---
        total_steps = (wait_steps + warmup_steps + active_steps) * repeat_cycles
        for i in range(total_steps):
            c = torch.matmul(a, b)
            d = torch.sin(c)
            e = torch.cos(d)
            f = torch.tan(e)
            g = torch.matmul(f, f)
            p.step()

extract_op_details([],"")