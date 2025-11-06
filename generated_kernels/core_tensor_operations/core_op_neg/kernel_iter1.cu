// [START kernel.cu]
#include <cuda_runtime.h>
#include <string>
#include <torch/extension.h> // Include PyTorch headers first to define c10::Half

// ============ DEVICE CODE ============

// CUDA kernel for element-wise negation
template <typename T>
__global__ void neg_kernel(const T* input, T* output, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Handle __half specifically for computation
        if constexpr (std::is_same<T, c10::Half>::value) {
            output[i] = static_cast<c10::Half>(-static_cast<float>(input[i]));
        } else {
            output[i] = -input[i];
        }
    }
}

// ============ HOST CODE ============

// Wrapper function for the negation operation
torch::Tensor launch_neg(torch::Tensor input) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device.");
    TORCH_CHECK(input.numel() > 0, "Input tensor must not be empty.");

    // Ensure tensor is contiguous for direct data access
    input = input.contiguous();

    // Get the total number of elements
    int64_t n_elements = input.numel();

    // Create an output tensor with the same shape and options as the input
    auto output = torch::empty_like(input);

    // Define grid and block dimensions
    int threads_per_block = 256;
    // Ensure blocks_per_grid is at least 1 for non-empty tensors
    int blocks_per_grid = (n_elements + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid == 0 && n_elements > 0) blocks_per_grid = 1; // Handle case where n_elements < threads_per_block

    // Dispatch to the correct kernel based on scalar type
    if (input.scalar_type() == torch::kFloat32) {
        neg_kernel<float><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n_elements
        );
    } else if (input.scalar_type() == torch::kFloat64) {
        neg_kernel<double><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<double>(),
            output.data_ptr<double>(),
            n_elements
        );
    } else if (input.scalar_type() == torch::kHalf) {
        neg_kernel<c10::Half><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for negation operation. Supported types are float32, float64, and half.");
    }

    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error in neg_kernel: ", cudaGetErrorString(err));
    }

    return output;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_neg, "Element-wise negation operation (CUDA)");
}
// [END kernel.cu]