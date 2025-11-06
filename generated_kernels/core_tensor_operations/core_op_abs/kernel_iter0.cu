// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string> // Required for std::string arguments if any, though not for this specific op.
#include <cmath>  // For std::abs

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise absolute value operation.
// Uses a grid-stride loop for efficient processing.
template <typename T>
__global__ void abs_kernel(const T* in, T* out, int64_t n_elements) {
    // Calculate global index using block and thread IDs
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // Calculate the total stride for the grid-stride loop
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Loop through elements with grid-stride
    for (int64_t i = idx; i < n_elements; i += stride) {
        // Handle half precision specifically, converting to float for computation
        // and back to c10::Half for storage.
        if constexpr (std::is_same_v<T, c10::Half>) {
            out[i] = static_cast<c10::Half>(fabsf(static_cast<float>(in[i])));
        } else {
            // For float and double, std::abs is sufficient.
            out[i] = static_cast<T>(std::abs(static_cast<double>(in[i])));
        }
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// C++ wrapper function for the absolute value operation.
// This function performs necessary checks, prepares tensors, and launches the CUDA kernel.
torch::Tensor launch_abs(torch::Tensor in_tensor) {
    // --- Input Tensor Checks ---
    TORCH_CHECK(in_tensor.is_cuda(), "Input tensor must be CUDA.");
    TORCH_CHECK(in_tensor.is_contiguous(), "Input tensor must be contiguous. Call .contiguous() on the tensor.");
    TORCH_CHECK(in_tensor.numel() > 0, "Input tensor must not be empty.");

    // Create an output tensor with the same shape and options as the input.
    auto out_tensor = torch::empty_like(in_tensor);

    // Get the total number of elements for kernel launch configuration.
    int64_t n_elements = in_tensor.numel();

    // --- CUDA Kernel Launch Configuration ---
    // Standard number of threads per block.
    const int threads_per_block = 256;
    // Calculate the number of blocks needed to cover all elements.
    const int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // --- Type Dispatch ---
    // Dispatch to the correct templated kernel based on the input tensor's scalar type.
    if (in_tensor.scalar_type() == torch::kFloat32) {
        abs_kernel<float><<<num_blocks, threads_per_block>>>(
            in_tensor.data_ptr<float>(),
            out_tensor.data_ptr<float>(),
            n_elements
        );
    } else if (in_tensor.scalar_type() == torch::kFloat64) {
        abs_kernel<double><<<num_blocks, threads_per_block>>>(
            in_tensor.data_ptr<double>(),
            out_tensor.data_ptr<double>(),
            n_elements
        );
    } else if (in_tensor.scalar_type() == torch::kHalf) {
        abs_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            in_tensor.data_ptr<c10::Half>(),
            out_tensor.data_ptr<c10::Half>(),
            n_elements
        );
    } else {
        // If an unsupported type is encountered, raise an error.
        TORCH_CHECK(false, "Unsupported scalar type for abs_kernel: ", in_tensor.scalar_type());
    }

    // --- CUDA Error Checking ---
    // Check for any CUDA errors that might have occurred during kernel launch.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    // Return the resulting tensor with absolute values.
    return out_tensor;
}

// ============ PYBIND11 MODULE DEFINITION ============
// This macro defines the Pybind11 module and exposes the C++ function to Python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Define the Python function "launch" which calls the C++ function "launch_abs".
    m.def("launch", &launch_abs, "Calculates the absolute value of elements in a PyTorch tensor (CUDA).");
}
// [END kernel.cu]