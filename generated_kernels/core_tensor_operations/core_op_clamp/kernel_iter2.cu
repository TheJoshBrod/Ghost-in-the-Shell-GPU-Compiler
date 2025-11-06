// [START kernel.cu]
#include <c10/util/Half.h>      // For c10::Half type used in host code
#include <ATen/Scalar.h>        // Explicitly for at::Scalar (c10::Scalar) definition
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/Dispatch.h>      // Required for AT_DISPATCH_FLOATING_TYPES_AND_HALF macros (if used)

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise clamp operation.
// It applies min/max clamping based on the `has_min` and `has_max` flags.
template <typename T>
__global__ void clamp_kernel(const T* input, T* output, bool has_min, T min_val, bool has_max, T max_val, int64_t n_elements) {
    // Calculate global index using a grid-stride loop
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        T value = input[i];
        
        // Apply minimum clamp if has_min is true
        if (has_min && value < min_val) {
            value = min_val;
        }
        // Apply maximum clamp if has_max is true
        if (has_max && value > max_val) {
            value = max_val;
        }
        output[i] = value;
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// C++ wrapper function to launch the CUDA clamp kernel.
// It handles tensor checks, optional scalar arguments, type dispatch, and kernel launch.
// The optional min/max arguments are now `c10::optional<at::Tensor>` to avoid pybind11 issues with c10::Scalar.
torch::Tensor launch_clamp(
    torch::Tensor self,
    c10::optional<at::Tensor> min_opt, // Changed from c10::optional<at::Scalar>
    c10::optional<at::Tensor> max_opt  // Changed from c10::optional<at::Scalar>
) {
    // 1. Critical checks for the input tensor
    TORCH_CHECK(self.is_cuda(), "Input tensor 'self' must be on CUDA.");
    TORCH_CHECK(self.is_contiguous(), "Input tensor 'self' must be contiguous.");

    // 2. Determine number of elements and create the output tensor
    int64_t n_elements = self.numel();
    torch::Tensor output = torch::empty_like(self);

    // 3. Process optional min/max scalar arguments
    bool has_min = min_opt.has_value();
    double min_val_d = 0.0;
    if (has_min) {
        // Extract scalar value from the tensor; PyTorch implicitly converts Python numbers to 0-dim tensors.
        min_val_d = min_opt.value().item<double>();
    }

    bool has_max = max_opt.has_value();
    double max_val_d = 0.0;
    if (has_max) {
        // Extract scalar value from the tensor.
        max_val_d = max_opt.value().item<double>();
    }

    // 4. Compute optimal grid and block sizes for the CUDA kernel launch
    int threads_per_block = 256;
    // Calculate number of blocks needed to cover all elements
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;
    // If n_elements is 0, num_blocks will be 0, and no kernel will be launched, which is correct.

    // 5. Manual type dispatch for different scalar types
    // Using AT_DISPATCH_FLOATING_TYPES_AND_HALF for a more robust approach.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "clamp_kernel", ([&] {
        // scalar_t is a type alias for the current dispatched type (e.g., float, double, c10::Half)
        scalar_t min_val_t = static_cast<scalar_t>(min_val_d);
        scalar_t max_val_t = static_cast<scalar_t>(max_val_d);

        clamp_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            self.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            has_min, min_val_t, has_max, max_val_t,
            n_elements
        );
    }));

    // 6. Check for any CUDA launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return output;
}

// Pybind11 module definition to expose the C++ function to Python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // The m.def signature must match the C++ function signature.
    m.def("launch", &launch_clamp, "Applies a clamp operation to the input tensor.");
}
// [END kernel.cu]