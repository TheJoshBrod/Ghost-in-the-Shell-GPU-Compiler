// [START kernel.cu]
#include <c10/util/Half.h>      // Required for c10::Half
#include <cuda_runtime.h>       // For CUDA Runtime API, __global__ keyword, and <<<>>> launch syntax
#include <type_traits>          // Required for std::is_same_v
#include <string>               // Required for std::string arguments if any, though not for sqrt

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise square root operation
template <typename T>
__global__ void sqrt_kernel(const T* input, T* output, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Handle different types for sqrt computation
        if constexpr (std::is_same_v<T, float>) {
            output[i] = sqrtf(input[i]);
        } else if constexpr (std::is_same_v<T, double>) {
            output[i] = sqrt(input[i]);
        } else if constexpr (std::is_same_v<T, c10::Half>) {
            // Convert half to float for computation, then back to half
            output[i] = static_cast<c10::Half>(sqrtf(static_cast<float>(input[i])));
        }
        // No else branch needed, AT_DISPATCH handles unsupported types at host level
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>    // Basic PyTorch extension headers
#include <ATen/Dispatch.h>      // Required for AT_DISPATCH_FLOATING_TYPES_AND_HALF

// Wrapper function to launch the sqrt CUDA kernel
torch::Tensor launch_sqrt(torch::Tensor input_tensor) {
    // --- Input Checks ---
    TORCH_CHECK(input_tensor.is_cuda(), "Input tensor must be on CUDA device.");
    TORCH_CHECK(input_tensor.numel() > 0, "Input tensor must not be empty.");
    
    // Ensure input is contiguous for direct pointer access
    input_tensor = input_tensor.contiguous();

    // --- Output Tensor Creation ---
    // Create an output tensor with the same shape and options (device, dtype) as the input
    torch::Tensor output_tensor = torch::empty_like(input_tensor);

    // --- Kernel Launch Parameters ---
    int64_t n_elements = input_tensor.numel();
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // --- Type Dispatch and Kernel Launch ---
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF handles float, double, and c10::Half types.
    // It automatically defines 'scalar_t' for the current dispatched type.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_tensor.scalar_type(), "sqrt_cuda", ([&] {
        sqrt_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input_tensor.data_ptr<scalar_t>(),
            output_tensor.data_ptr<scalar_t>(),
            n_elements
        );
    }));

    // --- CUDA Error Checking ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return output_tensor;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_sqrt, "Compute element-wise square root of a tensor on CUDA.");
}
// [END kernel.cu]