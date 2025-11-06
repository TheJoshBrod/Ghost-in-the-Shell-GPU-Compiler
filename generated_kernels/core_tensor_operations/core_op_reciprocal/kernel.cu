// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise reciprocal operation
template <typename T>
__global__ void reciprocal_kernel(const T* input, T* output, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Perform computation in float for better precision and handle half types
        float val = static_cast<float>(input[i]);
        output[i] = static_cast<T>(1.0f / val);
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for the reciprocal operation
torch::Tensor launch_reciprocal(torch::Tensor input) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device.");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous.");

    // Determine the number of elements
    int64_t n_elements = input.numel();

    // Create the output tensor with the same shape and options as the input
    auto output = torch::empty_like(input);

    // Calculate grid and block dimensions
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // Dispatch based on scalar type
    if (input.scalar_type() == torch::kFloat32) {
        reciprocal_kernel<float><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n_elements
        );
    } else if (input.scalar_type() == torch::kFloat64) {
        reciprocal_kernel<double><<<num_blocks, threads_per_block>>>(
            input.data_ptr<double>(),
            output.data_ptr<double>(),
            n_elements
        );
    } else if (input.scalar_type() == torch::kHalf) {
        reciprocal_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            input.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for reciprocal operation. Only float32, float64, and half are supported.");
    }

    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return output;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_reciprocal, "Computes the element-wise reciprocal of a tensor.");
}
// [END kernel.cu]