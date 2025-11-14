import torch
import json 
import re

aten_to_cuda = """
You are an expert CUDA and C++ developer specializing in PyTorch C++ extensions.

## YOUR TASK

Generate **one single compilable source file** (`kernel.cu`) for a given PyTorch operator. This file must contain:
1. CUDA kernel(s) (device code)
2. C++ wrapper function(s) (host code)
3. Pybind11 module definition

---

## OUTPUT FORMAT ⚠️ CRITICAL

You must output **exactly one code block** with these exact delimiters:

```cpp
// [START kernel.cu]
<your complete source code here>
// [END kernel.cu]
```

**If these delimiters are missing or the code is incomplete, the build will fail.**

---

## MANDATORY CODE STRUCTURE

```cpp
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

// ============ DEVICE CODE (CUDA kernels only) ============
// All __global__ kernels must be defined HERE, BEFORE PyTorch headers

template <typename T>
__global__ void my_kernel(...) {
    // kernel implementation
}

// ============ HOST CODE ============
#include <torch/extension.h>

// C++ wrapper functions go here
torch::Tensor launch_my_op(...) {
    // host code
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_my_op, "Operation description");
}
// [END kernel.cu]
```

---

## CRITICAL RULES ⛔

### What NOT to Include
1. ❌ NO `main()` function (this is a library, not a program)
2. ❌ NO `printf`, `std::cout`, or any I/O (verification happens in Python)
3. ❌ NO manual CUDA memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)
4. ❌ NO extra commentary outside the code block
5. ❌ NO multiple code blocks (exactly ONE block required)

### What to Include
1. ✅ All includes in the correct order
2. ✅ Template-based kernels for type flexibility
3. ✅ Proper error checking after kernel launches
4. ✅ Type dispatch for float32, float64, and half precision
5. ✅ Input validation and tensor checks
6. ✅ The Pybind11 function binding **must always be named "launch"**, regardless of the C++ wrapper function name.
7. ✅ Always use int64_t for all indices, dimensions, and sizes. Do not use int, int32_t, or any other type.
8. ✅ std::vector used for sizes must be std::vector<int64_t>.
9. ✅ When computing output shapes, always use std::vector<int64_t> and push_back int64_t dimensions.
10.✅ The C++ wrapper function can have any valid name, but the Pybind11 binding must call it with the name "launch".


---

## DEVICE CODE (CUDA Kernels)

### Requirements
- Place **before** `#include <torch/extension.h>`
- Must be templated: `template <typename T>`
- Use `int64_t` for all dimensions and indices
- For half precision: cast `T` to `float` for computation, cast back for storage

### Common Kernel Patterns

**Element-wise Operations (grid-stride loop):**
```cpp
template <typename T>
__global__ void elementwise_kernel(const T* input, T* output, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    
    for (int64_t i = idx; i < n_elements; i += stride) {
        float val = static_cast<float>(input[i]);
        // compute...
        output[i] = static_cast<T>(val);
    }
}
```

**Matrix Operations:**
```cpp
template <typename T>
__global__ void matrix_kernel(const T* A, const T* B, T* C, 
                               int64_t M, int64_t K, int64_t N) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    
    for (int64_t i = idx; i < total; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t row = i / N;
        int64_t col = i % N;
        // compute...
    }
}
```

---

## HOST CODE (C++ Wrapper)

### Function Signature
- Must **return** `torch::Tensor` (or `void` for in-place ops)
- Can accept mixed argument types:
  - `torch::Tensor` for tensor inputs
  - `int64_t`, `double`, `bool`, `std::string` for scalar parameters
  
### Required Validation (Tensors Only)

```cpp
torch::Tensor launch_my_op(torch::Tensor A, torch::Tensor B, 
                           int64_t param1, std::string mode) {
    // Check ONLY tensor arguments
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Type mismatch");
    
    // Make contiguous (avoids strided memory issues)
    A = A.contiguous();
    B = B.contiguous();
    
    // Extract dimensions
    int64_t M = A.size(0);
    int64_t N = A.size(1);
    
    // Allocate output
    auto C = torch::empty({M, N}, A.options());
    
    // Launch configuration
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    
    // Type dispatch
    if (A.scalar_type() == torch::kFloat32) {
        my_kernel<float><<<blocks, threads>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        my_kernel<double><<<blocks, threads>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), M, N
        );
    } else if (A.scalar_type() == torch::kHalf) {
        my_kernel<c10::Half><<<blocks, threads>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), C.data_ptr<c10::Half>(), M, N
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Supported: float32, float64, float16");
    }
    
    // CRITICAL: Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel failed: ", cudaGetErrorString(err));
    }
    
    return C;
}
```

### Common Patterns

**Grid/Block Sizing:**
- Standard: 256 threads per block
- Blocks: `(total_elements + 255) / 256`
- For 2D: can use `dim3 blocks(...)` if beneficial

**Type Dispatch:**
Always support at minimum: `kFloat32`, `kFloat64`, `kHalf`

**Output Allocation:**
- `torch::empty({shape}, input.options())` - matches input device/dtype
- `torch::zeros(...)` if initialization needed
- For in-place ops, modify input directly

---

## PYBIND11 BINDING

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_my_op, "Brief description of operation");
}
```

**Notes:**
- Pybind11 auto-detects argument types
- Use descriptive function names
- The function name "launch" is a convention but not required

---

## COMPLETE WORKING EXAMPLE

```cpp
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

// ============ DEVICE CODE ============

// Matrix multiplication: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
template <typename T>
__global__ void mm_kernel(const T* A, const T* B, T* C, 
                          int64_t M, int64_t K, int64_t N) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t total = M * N;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t row = i / N;
        int64_t col = i % N;
        
        float sum = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            sum += static_cast<float>(A[row * K + k]) * 
                   static_cast<float>(B[k * N + col]);
        }
        C[i] = static_cast<T>(sum);
    }
}

// ============ HOST CODE ============
#include <torch/extension.h>

torch::Tensor launch_mm(torch::Tensor A, torch::Tensor B) {
    // Validation
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions for mm");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Dtype mismatch");

    // Make contiguous
    A = A.contiguous();
    B = B.contiguous();

    // Dimensions
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    // Allocate output
    auto C = torch::empty({M, N}, A.options());

    // Launch config
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;

    // Type dispatch
    if (A.scalar_type() == torch::kFloat32) {
        mm_kernel<float><<<blocks, threads>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), 
            C.data_ptr<float>(), M, K, N
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        mm_kernel<double><<<blocks, threads>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), 
            C.data_ptr<double>(), M, K, N
        );
    } else if (A.scalar_type() == torch::kHalf) {
        mm_kernel<c10::Half><<<blocks, threads>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), 
            C.data_ptr<c10::Half>(), M, K, N
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }

    // Error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return C;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_mm, "Matrix multiplication (A @ B)");
}
// [END kernel.cu]
```

---

## CHECKLIST BEFORE RESPONDING

- [ ] Code block has `// [START kernel.cu]` and `// [END kernel.cu]`
- [ ] All includes in correct order
- [ ] CUDA kernels defined before `torch/extension.h`
- [ ] Kernels are templated with `<typename T>`
- [ ] Using `int64_t` for indices/dimensions
- [ ] Wrapper function returns `torch::Tensor`
- [ ] Tensor validation present (`.is_cuda()`, `.contiguous()`)
- [ ] Type dispatch for float32/float64/half
- [ ] `cudaGetLastError()` checked after kernel launch
- [ ] Pybind11 module defined
- [ ] No forbidden elements (main, printf, cudaMalloc, etc.)

---

## COMMON MISTAKES TO AVOID

1. **Forgetting error checks**: Always check `cudaGetLastError()`
2. **Wrong include order**: CUDA headers → kernels → PyTorch headers → host code
3. **Non-contiguous tensors**: Always call `.contiguous()`
4. **Integer overflow**: Use `int64_t`, not `int`
5. **Missing type casts**: Cast to float for computation with half precision
6. **Validating non-tensors**: Only check `.is_cuda()` on `torch::Tensor` args
7. **Using wrong binding name**: Must match the actual function name

---

When you receive an operator description, respond with ONLY the code block containing the complete, compilable kernel.cu file. No explanations, no markdown outside the code block.
```
"""

def generate_function_spec_from_calls(calls_dict, function_name):
    """
    Extract function specification from tracked PyTorch calls.
    Updated to support call records with 'args' and 'kwargs' instead of 'params'.
    """
    import torch
    
    if function_name not in calls_dict:
        return None
    
    call_list = calls_dict[function_name]
    if not call_list:
        return None
    
    ref_call = call_list[0]
    
    # Build combined parameter dict
    params = {}
    
    # Convert args to synthetic param names: arg0, arg1, ...
    for i, arg in enumerate(ref_call.get("args", [])):
        params[f"arg{i}"] = arg
    
    # Add kwargs normally
    params.update(ref_call.get("kwargs", {}))
    
    output = ref_call.get("output", None)
    
    param_specs = []
    for param_name, param_value in params.items():
        if isinstance(param_value, torch.Tensor):
            param_specs.append({
                "name": param_name,
                "type": "torch::Tensor",
                "dtype": str(param_value.dtype).replace('torch.', ''),
                "shape": list(param_value.shape),
                "description": f"Input tensor of shape {list(param_value.shape)}"
            })
        elif param_value is None:
            param_specs.append({
                "name": param_name,
                "type": "optional",
                "value": "None",
                "description": "Optional parameter (default: None)"
            })
        elif isinstance(param_value, bool):
            param_specs.append({
                "name": param_name,
                "type": "bool",
                "value": param_value,
                "description": "Boolean flag"
            })
        elif isinstance(param_value, int):
            param_specs.append({
                "name": param_name,
                "type": "int64_t",
                "value": param_value,
                "description": "Integer parameter"
            })
        elif isinstance(param_value, float):
            param_specs.append({
                "name": param_name,
                "type": "double",
                "value": param_value,
                "description": "Float parameter"
            })
        elif isinstance(param_value, str):
            param_specs.append({
                "name": param_name,
                "type": "std::string",
                "value": f'"{param_value}"',
                "description": "String parameter"
            })
        elif isinstance(param_value, (list, tuple)):
            param_specs.append({
                "name": param_name,
                "type": "std::vector<int64_t>",
                "value": list(param_value),
                "description": "List/tuple parameter"
            })
    
    # Output spec
    if isinstance(output, torch.Tensor):
        output_spec = {
            "type": "torch::Tensor",
            "dtype": str(output.dtype).replace('torch.', ''),
            "shape": list(output.shape),
            "description": f"Output tensor of shape {list(output.shape)}"
        }
    else:
        output_spec = {
            "type": str(type(output).__name__),
            "description": "Non-tensor output"
        }
    
    return {
        "function_name": function_name,
        "num_calls": len(call_list),
        "parameters": param_specs,
        "output": output_spec
    }


def format_operator_prompt(function_spec, profiler_context=None):
    """
    Format the function specification into a clear prompt for the LLM.
    This is what you append to the system prompt.
    
    Args:
        function_spec: Function specification dict
        profiler_context: Optional dict with {'aten_ops': [...], 'cuda_kernels': [...]}
    """
    
    prompt = f"""
## OPERATOR TO IMPLEMENT: {function_spec['function_name']}

### Function Signature

Based on {function_spec['num_calls']} tracked call(s), implement this operator:

**Parameters:**
"""
    
    # List all parameters
    for i, param in enumerate(function_spec['parameters'], 1):
        prompt += f"\n{i}. `{param['name']}` ({param['type']})"
        if 'shape' in param:
            prompt += f"\n   - Shape: {param['shape']}"
            prompt += f"\n   - dtype: {param['dtype']}"
        elif 'value' in param and param['value'] is not None:
            prompt += f"\n   - Default/Example: {param['value']}"
        prompt += f"\n   - {param['description']}"
    
    prompt += f"""

**Returns:**
- Type: {function_spec['output']['type']}
"""
    if 'shape' in function_spec['output']:
        prompt += f"- Shape: {function_spec['output']['shape']}\n"
        prompt += f"- dtype: {function_spec['output']['dtype']}\n"
    
    # Add profiler context if available
    if profiler_context:
        prompt += """

### Execution Context (from PyTorch Profiler)

This shows how PyTorch implements this operation internally:

"""
        if 'aten_ops' in profiler_context and profiler_context['aten_ops']:
            prompt += "**ATen Operations Called:**\n"
            for op in profiler_context['aten_ops']:
                prompt += f"- {op}\n"
            prompt += "\n"
        
        if 'cuda_kernels' in profiler_context and profiler_context['cuda_kernels']:
            prompt += "**CUDA Kernels Launched:**\n"
            for kernel in profiler_context['cuda_kernels']:
                prompt += f"- {kernel}\n"
            prompt += "\n"
        
        prompt += """**What This Means:**
- You may see setup operations (cudaGetDeviceCount, etc.) - ignore these
- Focus on the actual computation kernels
- Your implementation should replicate the core operation's behavior
- You don't need to match PyTorch's internal implementation exactly

"""
    
    # Add implementation guidance
    prompt += """
### Implementation Requirements

1. **C++ Wrapper Function Signature:**
   - Must accept ALL parameters listed above in the exact order
   - Optional parameters (None) should be handled with conditional logic
   - Return type must match the output specification

2. **CUDA Kernel:**
   - Must handle all tensor inputs
   - Compute the operation based on the function semantics
   - Support float32, float64, and half precision

3. **Parameter Handling:**
   - Validate tensor inputs (.is_cuda(), .contiguous())
   - Use scalar parameters directly in kernel launch
   - Handle optional tensors (check for null/validity)

Now generate the complete kernel.cu file following the system prompt guidelines.
"""
    
    return prompt


def parse_profiler_output(profiler_text):
    """
    Parse the profiler output string to extract ATen ops and CUDA kernels.
    
    Args:
        profiler_text: String containing profiler output with [Op: ...] and [Kernel: ...] lines
    
    Returns:
        dict with 'aten_ops' and 'cuda_kernels' lists
    """
    import re
    
    aten_ops = []
    cuda_kernels = []
    
    # Extract [Op: aten::something]
    op_pattern = r'\[Op:\s*(aten::\w+)\]'
    for match in re.finditer(op_pattern, profiler_text):
        aten_ops.append(match.group(1))
    
    # Extract [Kernel: something]
    kernel_pattern = r'\[Kernel:\s*([^\]]+)\]'
    for match in re.finditer(kernel_pattern, profiler_text):
        kernel_name = match.group(1).strip()
        # Filter out setup/instrumentation kernels
        if kernel_name not in ['Activity Buffer Request', 'Instrumentation', 'Resource']:
            cuda_kernels.append(kernel_name)
    
    return {
        'aten_ops': aten_ops,
        'cuda_kernels': cuda_kernels
    }


def generate_full_llm_prompt(calls_dict, function_name, profiler_output=None):
    """
    Complete pipeline: Generate the full prompt to send to an LLM.
    
    Args:
        calls_dict: Tracked function calls
        function_name: Function to implement (e.g., "torch.nn.functional.linear")
        profiler_output: Optional string output from PyTorch profiler
    
    Usage:
        calls = torch.load("tracked_calls.pt")
        
        # Without profiler context
        prompt = generate_full_llm_prompt(calls, "torch.nn.functional.linear")
        
        # With profiler context
        profiler_text = "..."  # Your aten/kernel output
        prompt = generate_full_llm_prompt(calls, "torch.nn.functional.linear", profiler_text)
    """
    
    # Extract function specification
    spec = generate_function_spec_from_calls(calls_dict, function_name)
    if spec is None:
        return f"Error: Could not generate spec for {function_name}"
    
    # Parse profiler output if provided
    profiler_context = None
    if profiler_output:
        profiler_context = parse_profiler_output(profiler_output)
    
    # Combine system prompt + operator specification
    full_prompt = format_operator_prompt(spec, profiler_context)
    
    return full_prompt
