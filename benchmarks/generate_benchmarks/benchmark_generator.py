import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from functools import wraps
import inspect

# from benchmarks.generate_benchmarks.trivial import SimpleModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SKIP_FUNCTIONS = [
    "has_torch_function",
    "handle_torch_function",
    "is_storage",
    "result_type",
    "get_default_dtype",
]

# ****************************
# Track specific PyTorch Calls
# ****************************
calls = {}
_wrapped = set()
ENABLE_WRAPPING = True

def wrap_function(module, func_name):
    if not ENABLE_WRAPPING:
        return
    func = getattr(module, func_name)
    if func in _wrapped:
        return
    _wrapped.add(func)

    module_path = module.__name__
    
    # Get function signature to extract default values
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        sig = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{module_path}.{func_name}"

        # Convert args to CPU tensors (lossless)
        ser_args = [
            a.detach().cpu() if isinstance(a, torch.Tensor) else a
            for a in args
        ]

        # Only record kwargs explicitly passed
        ser_kwargs = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }

        output = func(*args, **kwargs)

        if isinstance(output, torch.Tensor):
            ser_output = output.detach().cpu()
        elif isinstance(output, (list, tuple)):
            ser_output = [
                o.detach().cpu() if isinstance(o, torch.Tensor) else o
                for o in output
            ]
        else:
            ser_output = output

        calls.setdefault(key, []).append({
            "args": ser_args,
            "kwargs": ser_kwargs,
            "output": ser_output
        })

        return output

    setattr(module, func_name, wrapper)

# Wrap all functions in torch.nn.functional
for name in dir(F):
    if name.startswith("_"):
        continue  # skip private names like _in_projection, etc.
    if any(skip in name for skip in ["torch_function", "storage", "result_type", "dtype"]):
        continue  # skip helpers by substring match
    obj = getattr(F, name)
    if callable(obj):
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
