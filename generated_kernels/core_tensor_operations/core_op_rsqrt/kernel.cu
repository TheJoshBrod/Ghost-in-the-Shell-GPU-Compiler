// [START kernel.cu]
#include <cuda_runtime.h> // Sufficient, cuda.h not strictly necessary
#include <cmath>          // For sqrtf and sqrt
#include <type_traits>    // Required for std::is_same
#include <c10/util/Half.h> // Required for c10::Half type

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise reciprocal square root (rsqrt)
// Computes output[i] = 1.0 / sqrt(input[i]) for each element.
template <typename T>
__global__ void rsqrt_kernel(const T* input, T* output, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // The host-side dispatch ensures T is one of float, double, or c10::Half.
        // Using if constexpr for type-dependent compilation is generally preferred
        // but std::is_same combined with type-specific casts/functions works too.
        if (std::is_same<T, c10::Half>::value) {
            // Convert c10::Half to float for computation, then back
            float val = static_cast<float>(input[i]);
            output[i] = static_cast<T>(1.0f / sqrtf(val));
        } else if (std::is_same<T, float>::value) {
            // Use single precision sqrtf for float inputs
            output[i] = static_cast<T>(1.0f / sqrtf(input[i]));
        } else if (std::is_same<T, double>::value) {
            // Use double precision sqrt for double inputs
            output[i] = static_cast<T>(1.0 / sqrt(input[i]));
        }
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// C++ wrapper function to launch the rsqrt CUDA kernel
torch::Tensor launch_rsqrt(torch::Tensor input) {
    // Critical checks for tensor arguments
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on a CUDA device.");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous.");
    TORCH_CHECK(input.dim() >= 0, "Input tensor must have dimensions."); // Can be scalar (0-dim)

    // Create an output tensor with the same shape and options as the input
    auto output = torch::empty_like(input);

    int64_t n_elements = input.numel();
    if (n_elements == 0) {
        return output; // Return empty tensor if input is empty
    }

    // Standard CUDA kernel launch configuration
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // Manual type dispatch based on input tensor's scalar type
    if (input.scalar_type() == torch::kFloat32) {
        rsqrt_kernel<float><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n_elements
        );
    } else if (input.scalar_type() == torch::kFloat64) {
        rsqrt_kernel<double><<<num_blocks, threads_per_block>>>(
            input.data_ptr<double>(),
            output.data_ptr<double>(),
            n_elements
        );
    } else if (input.scalar_type() == torch::kHalf) {
        rsqrt_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            input.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            n_elements
        );
    } else {
        // Error for unsupported tensor types
        TORCH_CHECK(false, "Unsupported scalar type for rsqrt: ", input.scalar_type());
    }

    // Check for any CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return output;
}

// Pybind11 module definition to expose the C++ function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_rsqrt, "Calculates the reciprocal of the square root of the input tensor elements.");
}
// [END kernel.cu]