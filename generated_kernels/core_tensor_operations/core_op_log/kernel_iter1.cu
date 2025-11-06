// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cmath> // For std::log, std::logf

// PyTorch headers for host code and type dispatch
#include <torch/extension.h>
#include <ATen/Dispatch.h> // Required for AT_DISPATCH_FLOATING_TYPES_AND_HALF
#include <c10/util/Half.h> // Required for c10::Half

// ============ DEVICE CODE ============

// CUDA kernel for element-wise logarithm operation
template <typename T>
__global__ void log_kernel(const T* input, T* output, int64_t num_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < num_elements; i += stride) {
        // Perform logarithm operation based on type
        // c10::Half implicitly converts to float, logf returns float, then converts back to T
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, c10::Half>) {
            output[i] = static_cast<T>(logf(static_cast<float>(input[i])));
        } else if constexpr (std::is_same_v<T, double>) {
            output[i] = static_cast<T>(log(static_cast<double>(input[i])));
        }
        // No need for other types as AT_DISPATCH_FLOATING_TYPES_AND_HALF only covers these.
    }
}

// ============ HOST CODE ============

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

    // Dispatch to the appropriate kernel based on scalar type using AT_DISPATCH macro
    // This macro will define 'scalar_t' based on the input tensor's data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "log_kernel", [&] {
        log_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements
        );
    });

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