// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>  // For sinf and sin
#include <c10/util/Half.h> // Required for c10::Half type
#include <torch/extension.h>

// CUDA kernel for element-wise sine operation
template <typename T>
__global__ void sin_kernel(const T* input, T* output, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // For half precision, cast to float for computation and then back.
        if constexpr (std::is_same_v<T, c10::Half>) {
            output[i] = static_cast<T>(sinf(static_cast<float>(input[i])));
        } else if constexpr (std::is_same_v<T, float>) {
            output[i] = sinf(input[i]);
        } else if constexpr (std::is_same_v<T, double>) {
            output[i] = sin(input[i]);
        }
    }
}

// Wrapper function for the sine operation
torch::Tensor launch_sin(torch::Tensor input) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device.");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous.");

    // Get number of elements
    int64_t n_elements = input.numel();

    // Create output tensor with the same shape and options as the input
    auto output = torch::empty_like(input);

    // Determine grid and block sizes
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;
    // Cap blocks to avoid potential issues with very large numbers, though typically handled by runtime.
    // Max grid size in x-dimension is 2^31-1 for modern GPUs, so no overflow here for typical tensor sizes.

    // Manual type dispatch
    if (input.scalar_type() == torch::kFloat32) {
        sin_kernel<float><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), n_elements
        );
    } else if (input.scalar_type() == torch::kFloat64) {
        sin_kernel<double><<<num_blocks, threads_per_block>>>(
            input.data_ptr<double>(), output.data_ptr<double>(), n_elements
        );
    } else if (input.scalar_type() == torch::kHalf) {
        sin_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            input.data_ptr<c10::Half>(), output.data_ptr<c10::Half>(), n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for sin operation: ", input.scalar_type());
    }

    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch error: ", cudaGetErrorString(err));
    }

    return output;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_sin, "Element-wise sine operation (CUDA)");
}
// [END kernel.cu]