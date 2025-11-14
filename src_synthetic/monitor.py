"""Monitors and Preprocesses PyTorch Aten and CUDA Kernel Abstraction Layer Calls for Generator"""
import torch

aten_output: str = ""
kernel_output: str = ""

if not torch.cuda.is_available():
    raise RuntimeError("This example requires a CUDA-enabled PyTorch installation.")

def handle_trace(prof):
    """
    Display profiled events grouped by high-level ATen operators, 
    with the associated CUDA kernels underneath, preserving order.
    """
    global kernel_output
    global aten_output

    current_op = None
    for event in prof.events():
        if event.self_device_time_total == 0 and event.self_cpu_time_total == 0:
            continue  # skip events with no device time (profiler noise)

        if event.key.startswith("aten::"):
            # Start a new high-level op
            current_op = event.key
            aten_output += f"[Op: {current_op}]\n"
            if event.input_shapes:
                aten_output += f"  Inputs: {event.input_shapes}\n"
            else:
                aten_output += f"  Inputs: (Not available)\n"
            aten_output += f"  Device Time (ms): {event.self_device_time_total / 1000:.3f}\n"

        elif "ProfilerStep" not in event.key:
            # Low-level kernel; associate with current op if exists
            if current_op:
                kernel_output += f"    [Kernel: {event.key}]\n"
                kernel_output += f"      Device Time (ms): {event.self_device_time_total / 1000:.3f}\n"
            else:
                # Kernel not associated with any high-level op
                kernel_output += f"[Kernel: {event.key}]\n"
                kernel_output += f"  Device Time (ms): {event.self_device_time_total / 1000:.3f}\n"


def profile_single_op(context: dict, full_exec_string: str) -> tuple[torch.Tensor, str]:
    """
    Profiles a *single line* of PyTorch code and returns its output tensor
    and the formatted op_details string.
    
    Args:
        context (dict): The current execution context containing defined variables (like 'a' and 'b').
        full_exec_string (str): The single line of code to execute (e.g., "c = torch.matmul(a, b)").
        
    Output:
        ground_truth_tensor (torch.Tensor): The resulting tensor (e.g., 'c').
        op_details (str): Formatted string of Aten/kernel info for the LLM.
    """
    
    # 1. Reset global profiler strings for this specific op
    global aten_output, kernel_output
    aten_output = ""
    kernel_output = ""
    
    # 2. Set deterministic state
    torch.cuda.manual_seed(100)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3. We'll run the profiler for just a few steps on this one op
    # (wait, warmup, active)
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1)
    
    # 4. Create a *copy* of the context to safely execute the line
    # This ensures we don't modify the main context if exec fails
    # (though main.py will update its own context upon success)
    temp_context = context.copy()
    
    # 5. Get the name of the variable being assigned
    # e.g., "c = torch.matmul(a, b)" -> "c"
    try:
        assignment_name = full_exec_string.split('=')[0].strip()
    except Exception:
        raise ValueError(f"Operation '{full_exec_string}' is not a valid assignment.")

    # 6. Run the profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=handle_trace,
        record_shapes=True
    ) as p:
        for _ in range(3): # Total steps: wait + warmup + active
            print(full_exec_string)
            exec(full_exec_string, temp_context)
            p.step()

    # 7. Get the resulting tensor from the temporary context
    ground_truth_tensor = temp_context.get(assignment_name)
    if ground_truth_tensor is None or not isinstance(ground_truth_tensor, torch.Tensor):
        raise ValueError(f"Failed to get output tensor '{assignment_name}' from context after exec.")

    # 8. Format op_details string for the LLM
    op_details = f"aten output:\n{aten_output}\n\n\nkernel output:\n{kernel_output}"
    print(op_details)
    return ground_truth_tensor, op_details