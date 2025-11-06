// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cmath> // For std::log, std::logf

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise logarithm operation
template <typename T>
__global__ void log_kernel(const T* input, T* output, int64_t num_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < num_elements; i += stride) {
        // Perform logarithm operation based on type
        // Use float for __half computation
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, c10::Half>) {
            output[i] = static_cast<T>(logf(static_cast<float>(input[i])));
        } else if constexpr (std::is_same_v<T, double>) {
            output[i] = static_cast<T>(log(static_cast<double>(input[i])));
        }
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// C++ wrapper function to launch the CUDA kernel
torch::Tensor launch_log(torch::Tensor input) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device.");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous.");
    TORCH_CHECK(input.numel() > 0, "Input tensor must not be empty.");

    // Create output tensor with the same shape and options as the input
    torch::Tensor output = torch::empty_like(input);

    int64_t num_elements = input.numel();

    // Configure kernel launch parameters
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Dispatch to the appropriate kernel based on scalar type
    if (input.scalar_type() == torch::kFloat32) {
        log_kernel<float><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), num_elements
        );
    } else if (input.scalar_type() == torch::kFloat64) {
        log_kernel<double><<<num_blocks, threads_per_block>>>(
            input.data_ptr<double>(), output.data_ptr<double>(), num_elements
        );
    } else if (input.scalar_type() == torch::kHalf) {
        log_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            input.data_ptr<c10::Half>(), output.data_ptr<c10::Half>(), num_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type for log operation: ", input.scalar_type());
    }

    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error for log operation: ", cudaGetErrorString(err));
    }

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_log, "Element-wise log operation on a tensor.");
}
// [END kernel.cu]