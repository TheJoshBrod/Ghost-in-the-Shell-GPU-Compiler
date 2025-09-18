import torch
import threading
import queue
import time

# Check hardware if CUDA 
if not torch.cuda.is_available():
    raise RuntimeError("This example requires a CUDA-enabled PyTorch installation.")

# A thread-safe queue to pass messages
kernel_queue = queue.Queue()

def print_kernel_info(kernel) -> None: 
    """Outputs kernel event to commandline."""

    stats = "[Monitor Thread]\n"
    stats += f"\t- name: {kernel.name}\n"
    stats += f"\t- CUDA time: {kernel.device_time_total}\n"
    print(stats)

def monitor_thread() -> None:
    """Waits for CUDA kernel events from the queue and prints them."""

    print("[Monitor Thread] Started.")
    while True:
        kernel = kernel_queue.get() # Blocking func call
        if kernel is None:
            print("[Monitor Thread] Exiting.")
            break
        print_kernel_info(kernel)

def handle_trace(prof: torch.profiler.profile) -> None:
    """
    Profiler callback that filters for CUDA kernel events and sends
    them to the monitor thread via the queue.
    """
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernel_queue.put(event)

def main() -> None:
    print(f"[Main Thread] Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Start the monitor thread
    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()

    # Create dummy tensors on the GPU
    a = torch.randn(2048, 2048, device='cuda')
    b = torch.randn(2048, 2048, device='cuda')
    
    
    # Execute PyTorch Code w/ profiler
    print("\n[Main Thread] Starting profiled execution...")
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=5)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=handle_trace
    ) as p:
        # Run a loop of GPU work. The profiler will trace it in chunks.
        for i in range(200):
            c = torch.matmul(a, b)
            d = torch.sin(c)
            p.step() # Tells profiler step was completed
            if i % 20 == 0:
                 time.sleep(1)


    # Signal the monitor thread to stop and wait for it
    print("\n[Main Thread] Profiled execution finished.")
    kernel_queue.put(None)
    monitor.join()

if __name__ == "__main__":
    main()