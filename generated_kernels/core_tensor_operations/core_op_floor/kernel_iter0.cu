// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string> // Required for std::string arguments if used, though not for aten::floor

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise floor operation
template <typename T>
__global__ void floor_kernel(const T* input, T* output, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // For float/double, use respective floor functions.
        // For half, cast to float, apply floorf, then cast back.
        if constexpr (std::is_same_v<T, float>) {
            output[i] = floorf(input[i]);
        } else if constexpr (std::is_same_v<T, double>) {
            output[i] = floor(input[i]);
        } else if constexpr (std::is_same_v<T, __half>) {
            output[i] = static_cast<__half>(floorf(static_cast<float>(input[i])));
        } else {
            // This path should ideally not be taken due to type dispatch in host code,
            // but good practice to handle.
            output[i] = input[i]; // Fallback, but incorrect for other types
        }
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for the floor operation
torch::Tensor launch_floor(torch::Tensor input) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(input.is_floating_point(), "Input tensor must be of a floating point type.");

    // Ensure the input is contiguous for simpler kernel access
    input = input.contiguous();

    // Create the output tensor with the same shape and options as the input
    auto output = torch::empty_like(input);

    // Get the total number of elements
    int64_t n_elements = input.numel();

    // Configure kernel launch parameters
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // Dispatch based on scalar type
    if (input.scalar_type() == torch::kFloat32) {
        floor_kernel<float><<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), n_elements
        );
    } else if (input.scalar_type() == torch::kFloat64) {
        floor_kernel<double><<<num_blocks, threads_per_block>>>(
            input.data_ptr<double>(), output.data_ptr<double>(), n_elements
        );
    } else if (input.scalar_type() == torch::kHalf) {
        floor_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            input.data_ptr<c10::Half>(), output.data_ptr<c10::Half>(), n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for floor operation: ", input.scalar_type());
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
    m.def("launch", &launch_floor, "Applies the floor function element-wise to a tensor.");
}
// [END kernel.cu]