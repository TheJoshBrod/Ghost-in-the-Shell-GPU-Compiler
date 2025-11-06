// [START kernel.cu]
#include <c10/util/Half.h>      // For c10::Half type used in host code
#include <ATen/Scalar.h>        // Explicitly for at::Scalar (c10::Scalar) definition
#include <cuda.h>
#include <cuda_runtime.h>

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
torch::Tensor launch_clamp(
    torch::Tensor self,
    c10::optional<at::Scalar> min_scalar,
    c10::optional<at::Scalar> max_scalar
) {
    // 1. Critical checks for the input tensor
    TORCH_CHECK(self.is_cuda(), "Input tensor 'self' must be on CUDA.");
    TORCH_CHECK(self.is_contiguous(), "Input tensor 'self' must be contiguous.");

    // 2. Determine number of elements and create the output tensor
    int64_t n_elements = self.numel();
    torch::Tensor output = torch::empty_like(self);

    // 3. Process optional min/max scalar arguments
    bool has_min = min_scalar.has_value();
    // Use 0.0 as a placeholder if not present; its value will only be used if has_min is true.
    double min_val_d = has_min ? min_scalar.value().to<double>() : 0.0;

    bool has_max = max_scalar.has_value();
    // Use 0.0 as a placeholder if not present; its value will only be used if has_max is true.
    double max_val_d = has_max ? max_scalar.value().to<double>() : 0.0;

    // 4. Compute optimal grid and block sizes for the CUDA kernel launch
    int threads_per_block = 256;
    // Calculate number of blocks needed to cover all elements
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;
    // If n_elements is 0, num_blocks will be 0, and no kernel will be launched, which is correct.

    // 5. Manual type dispatch for different scalar types
    if (self.scalar_type() == torch::kFloat32) {
        float min_val_f = static_cast<float>(min_val_d);
        float max_val_f = static_cast<float>(max_val_d);
        clamp_kernel<float><<<num_blocks, threads_per_block>>>(
            self.data_ptr<float>(), output.data_ptr<float>(),
            has_min, min_val_f, has_max, max_val_f,
            n_elements
        );
    } else if (self.scalar_type() == torch::kFloat64) {
        clamp_kernel<double><<<num_blocks, threads_per_block>>>(
            self.data_ptr<double>(), output.data_ptr<double>(),
            has_min, min_val_d, has_max, max_val_d,
            n_elements
        );
    } else if (self.scalar_type() == torch::kHalf) {
        c10::Half min_val_h = static_cast<c10::Half>(min_val_d);
        c10::Half max_val_h = static_cast<c10::Half>(max_val_d);
        clamp_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            self.data_ptr<c10::Half>(), output.data_ptr<c10::Half>(),
            has_min, min_val_h, has_max, max_val_h,
            n_elements
        );
    } else {
        // Raise an error for unsupported data types
        TORCH_CHECK(false, "Unsupported scalar type for clamp operation: ", self.scalar_type());
    }

    // 6. Check for any CUDA launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return output;
}

// Pybind11 module definition to expose the C++ function to Python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_clamp, "Applies a clamp operation to the input tensor.");
}
// [END kernel.cu]