import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
import json
from functools import wraps

# from benchmarks.generate_benchmarks.trivial import SimpleModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ****************************
# Track specific PyTorch Calls
# ****************************
calls = {}
_wrapped = set()
ENABLE_WRAPPING = True
MAX_CALLS_PER_FUNC = 1e10
def wrap_function(module, func_name):
    if not ENABLE_WRAPPING:
        return      
    func = getattr(module, func_name)
    if func in _wrapped:
        return
    _wrapped.add(func)

    module_path = module.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Only track first n times
        key = f"{module_path}.{func_name}"
        if len(calls.get(key, [])) >= MAX_CALLS_PER_FUNC:
            return func(*args, **kwargs)

        # Convert arguments to JSON-serializable format
        arg_repr = [arg.detach().cpu() if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwarg_repr = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}


        # Store in calls dictionary
        key = f"{module_path}.{func_name}"
        if key not in calls:
            calls[key] = []

        # # Optional print for live feedback
        # print(f"[CALL] {key} args={arg_repr} kwargs={kwarg_repr}")

        # Call the original function
        output = func(*args, **kwargs)

        # Convert output to JSON-serializable format
        if isinstance(output, torch.Tensor):
            output_repr = output.detach().cpu()
        elif isinstance(output, (list, tuple)):
            output_repr = [o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output]
        else:
            output_repr = output

        # Append args, kwargs, and output to tracker
        calls[key].append({"args": arg_repr, "kwargs": kwarg_repr, "output": output_repr})

        return output

    setattr(module, func_name, wrapper)

# Wrap all functions in torch.nn.functional
for name in dir(F):
    if callable(getattr(F, name)):
        wrap_function(F, name)


# ****************************
#   Initialize PyTorch Model
# ****************************

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
inputs = tokenizer("Hello, this is a test of your tracking code!", return_tensors="pt")


# ****************************
#        ATen Profiler 
# ****************************

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
with profiler.profile(record_shapes=True, use_device=device_str) as prof:
    with profiler.record_function("forward"):
        with torch.no_grad():
            outputs = model(**inputs)
print(prof.key_averages().table(sort_by="count", row_limit=50))

# ****************************
#  Export PyTorch API In/Out 
# ****************************

print("Saving....")
torch.save(calls, f"benchmarks/generate_benchmarks/{model.__class__.__name__}.pt")
print("Saved to output.pt")