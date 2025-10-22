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
        if event.self_device_time_total == 0:
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


def extract_op_details(program: list[str]):
    """Extracts relevant information of sample PyTorch code including: 
    - High level kernel PyTorch operators
    - Names of PyTorch’s internal CUDA kernel/cuBLAS kernels/etc.
    - Correct output given the input
    
    Args:
        inputs (list[str]): List of inputs that the sample takes (values, size, etc.)
        code (str): Literal PyTorch code to be executed
    
    Output:
        str: Correct value of the PyTorch sample
        str: Formatted string of high Aten representation of sample
        str: Formatted string of low level kernel names of sample
    """

    # Set state to be deterministic
    torch.cuda.manual_seed(100)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("\n[Main Thread] Starting profiled execution...")
    
    context = {"torch": torch}

    # execute definitions
    for definition in program["definitions"]:
        var = definition["variable"]
        val = definition["value"]
        exec(f"{var} = {val}", context)

    print("\n[Main Thread] Executed definitions...")
    
    wait_steps = 1
    warmup_steps = 1
    active_steps = 2
    repeat_cycles = 1

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
            
            for op in program["operations"]:
                assignment = op["assignment"]
                operation = op["operation"]
                exec(f"{assignment} = {operation}", context)
            p.step()


        # TODO:
        # I put this here to guarantee profiler is done before continuing
        # It turns the execution to be synchronous, I don't think it breaks anything but should be looked into.  
        p.stop()
        handle_trace(p)

        # TODO: Get output variable?
        output = context.get("output", None)

    return output, aten_output, kernel_output
