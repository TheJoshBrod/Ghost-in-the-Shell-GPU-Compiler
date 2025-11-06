// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath> // For std::exp and std::expf
#include <c10/util/Half.h> // Required for c10::Half type
#include <torch/extension.h> // Include PyTorch extension header here

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise exponential function
template <typename T>
__global__ void exp_kernel(const T* input, T* output, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Handle half precision separately for computation accuracy
        if constexpr (std::is_same_v<T, c10::Half>) {
            float val_float = static_cast<float>(input[i]);
            output[i] = static_cast<c10::Half>(expf(val_float));
        } else if constexpr (std::is_same_v<T, float>) {
            output[i] = expf(input[i]);
        } else if constexpr (std::is_same_v<T, double>) {
            output[i] = exp(input[i]);
        }
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============

// C++ wrapper function to launch the CUDA kernel
torch::Tensor launch_exp(torch::Tensor input_tensor) {
    // 1. Input checks
    TORCH_CHECK(input_tensor.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(input_tensor.is_contiguous(), "Input tensor must be contiguous.");
    TORCH_CHECK(
        input_tensor.scalar_type() == torch::kFloat32 ||
        input_tensor.scalar_type() == torch::kFloat64 ||
        input_tensor.scalar_type() == torch::kHalf,
        "Unsupported data type. Only float32, float64, and half are supported."
    );

    // 2. Determine output tensor properties and create it
    auto output_tensor = torch::empty_like(input_tensor);
    int64_t n_elements = input_tensor.numel();

    // 3. Configure kernel launch parameters
    const int threads_per_block = 256;
    // Fix: Corrected typo from 'threads_per_per_block' to 'threads_per_block'
    const int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // 4. Dispatch to the correct kernel based on data type
    if (input_tensor.scalar_type() == torch::kFloat32) {
        exp_kernel<float><<<num_blocks, threads_per_block>>>(
            input_tensor.data_ptr<float>(),
            output_tensor.data_ptr<float>(),
            n_elements
        );
    } else if (input_tensor.scalar_type() == torch::kFloat64) {
        exp_kernel<double><<<num_blocks, threads_per_block>>>(
            input_tensor.data_ptr<double>(),
            output_tensor.data_ptr<double>(),
            n_elements
        );
    } else if (input_tensor.scalar_type() == torch::kHalf) {
        exp_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            input_tensor.data_ptr<c10::Half>(),
            output_tensor.data_ptr<c10::Half>(),
            n_elements
        );
    } else {
        // This case should ideally be caught by TORCH_CHECK above, but good for safety.
        TORCH_CHECK(false, "Unsupported scalar type for exp operation.");
    }

    // 5. Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output_tensor;
}

// ============ PYBIND11 MODULE DEFINITION ============
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_exp, "Element-wise exponential CUDA operation.");
}
// [END kernel.cu]