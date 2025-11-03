// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cmath> // For std::sin

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise sine operation
template <typename T>
__global__ void sin_kernel(const T* input, T* output, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Use float for computation to leverage `sinf` for performance, then cast back.
        // This also handles __half transparently by casting to float first.
        output[i] = static_cast<T>(sinf(static_cast<float>(input[i])));
    }
}

// Specialization for double to use sin directly
template <>
__global__ void sin_kernel<double>(const double* input, double* output, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        output[i] = sin(input[i]);
    }
}


// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function to launch the sin kernel
torch::Tensor launch_sin(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    // Get the total number of elements in the tensor
    int64_t n_elements = input.numel();

    // Create an output tensor with the same properties as the input
    auto output = torch::empty_like(input);

    // Define kernel launch parameters
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // Dispatch based on scalar type
    if (input.scalar_type() == torch::kFloat32) {
        sin_kernel<float><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n_elements
        );
    } else if (input.scalar_type() == torch::kFloat64) {
        sin_kernel<double><<<num_blocks, threads_per_block>>>(
            input.data_ptr<double>(),
            output.data_ptr<double>(),
            n_elements
        );
    } else if (input.scalar_type() == torch::kHalf) {
        sin_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            input.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for sin operation.");
    }

    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_sin, "Sine CUDA kernel");
}
// [END kernel.cu]