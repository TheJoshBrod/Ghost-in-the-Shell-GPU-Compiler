// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cmath> // For std::round
#include <c10/util/Half.h> // Required for c10::Half

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise rounding
template <typename T>
__global__ void round_kernel(const T* input, T* output, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // For half precision, cast to float, round, then cast back
        // For float/double, std::round will work directly or through casting
        if constexpr (std::is_same_v<T, c10::Half>) {
            float val_float = static_cast<float>(input[i]);
            output[i] = static_cast<c10::Half>(std::round(val_float));
        } else {
            output[i] = static_cast<T>(std::round(input[i]));
        }
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for the round operation
torch::Tensor launch_round(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous.");

    // Create an output tensor with the same shape and options as the input
    auto output = torch::empty_like(input);

    int64_t n_elements = input.numel();

    // Standard block size and calculate grid size
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;
    if (num_blocks == 0) num_blocks = 1; // Ensure at least one block for empty tensors

    // Type dispatch for kernel launch
    if (input.scalar_type() == torch::kFloat32) {
        round_kernel<float><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n_elements
        );
    } else if (input.scalar_type() == torch::kFloat64) {
        round_kernel<double><<<num_blocks, threads_per_block>>>(
            input.data_ptr<double>(),
            output.data_ptr<double>(),
            n_elements
        );
    } else if (input.scalar_type() == torch::kHalf) {
        round_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            input.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for round operation: ", input.scalar_type());
    }

    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error in round_kernel: ", cudaGetErrorString(err));
    }

    return output;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_round, "Rounds elements of a tensor.");
}
// [END kernel.cu]