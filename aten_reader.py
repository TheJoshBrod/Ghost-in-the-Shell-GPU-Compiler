import torch
import threading
import queue
import time

# Check hardware if CUDA 
if not torch.cuda.is_available():
    raise RuntimeError("This example requires a CUDA-enabled PyTorch installation.")

# A thread-safe queue to pass messages
op_queue = queue.Queue()

def monitor_thread():
    """
    This function runs in a separate thread.
    It waits for operation names to appear in the queue and prints them.
    """
    print("[Monitor Thread] Started.")
    while True:
        op_name = op_queue.get()
        if op_name is None:
            print("[Monitor Thread] Exiting.")
            break   
        print(f"[Monitor Thread] {op_name}")

class OpMonitor(torch.utils._python_dispatch.TorchDispatchMode):
    """
    When this context manager is active, any PyTorch operation will
    first call this __torch_dispatch__ method.
    """
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        
        # Send the operation name to the monitor thread
        op_name = func.__name__
        op_queue.put(op_name)
        
        return func(*args, **kwargs)

def main() -> None:
    print(f"[Main Thread] Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Start the monitor thread
    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()

    # Create dummy tensors on the GPU
    a = torch.randn(10, 10, device='cuda')
    b = torch.randn(10, 10, device='cuda')
    
    print("\n[Main Thread] Enabling dispatch hook and running PyTorch code...")
    with OpMonitor():
        # Any operation inside this block will be intercepted
        d = torch.sin(a)
        time.sleep(0.5)
        e = torch.matmul(d, b)
        time.sleep(0.5)
        f = e.relu()

    print("\n[Main Thread] PyTorch code finished.")
    
    # Signal the monitor thread to stop and wait for it to finish
    op_queue.put(None)
    monitor.join()

if __name__ == "__main__":
    main()