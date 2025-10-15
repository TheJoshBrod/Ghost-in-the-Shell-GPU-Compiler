import torch
import time

# ---- Check CUDA ----
if not torch.cuda.is_available():
    raise RuntimeError("This example requires a CUDA-enabled PyTorch installation.")

# ---------- CORRECTED Profiler Callback ----------
def handle_trace(prof):
    print("\n--- Profiler Trace Ready ---")

    # Filter for events that actually ran on the CUDA device
    # by checking if they have a recorded CUDA time.
    device_events = [e for e in prof.events() if e.device_time > 0]
    device_events.sort(key=lambda e: e.time_range.start)

    print(f"Found {len(device_events)} CUDA kernel events.")
    for event in device_events:
        # The CPU operator that launched this kernel
        parent_op = event.cpu_parent
        
        if parent_op:
            print("[Op + Kernel]")
            print(f"  op_name: {parent_op.name}")
            print(f"  kernel_name: {event.name}")
            print(f"  Device time (us): {event.cuda_time}")
        else:
            print("[Kernel Event, no op match]")
            print(f"  kernel_name: {event.name}")
            print(f"  Device time (us): {event.device_time}")
            
    print("--------------------------\n")

# ---------- Main Execution Logic (Unchanged) ----------
def main():
    print(f"[Main Thread] Using GPU: {torch.cuda.get_device_name(0)}")

    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")

    print("\n[Main Thread] Starting profiled execution...")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=2)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=handle_trace
    ) as p:
        for i in range(100): # 1 wait + 1 warmup + 2 active = 4 steps
            c = torch.matmul(a, b)
            d = torch.sin(c)

            
    print("\n[Main Thread] Execution finished.")

if __name__ == "__main__":
    main()