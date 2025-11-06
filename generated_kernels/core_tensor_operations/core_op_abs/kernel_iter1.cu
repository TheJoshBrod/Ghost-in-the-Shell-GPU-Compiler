// [START kernel.cu]
#include <torch/extension.h> // Provides c10::Half, other PyTorch types, and general utilities
#include <c10/util/Half.h>   // Explicitly include for c10::Half type definition
#include <cuda_runtime.h>    // For CUDA API calls (e.g., cudaGetLastError) and types (e.g., cudaError_t)
#include <cmath>             // For standard math functions like std::abs, though fabsf/fabs are usually preferred in device code
#include <type_traits>       // For std::is_same_v

// ============ DEVICE CODE ============

// CUDA kernel for element-wise absolute value operation.
// Uses a grid-stride loop for efficient processing across potentially many elements.
template <typename T>
__global__ void abs_kernel(const T* in, T* out, int64_t n_elements) {
    // Calculate global index using block and thread IDs
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // Calculate the total stride for the grid-stride loop (total threads in the grid)
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Loop through elements with grid-stride to ensure all elements are processed
    // regardless of the number of threads or blocks.
    for (int64_t i = idx; i < n_elements; i += stride) {
        // Use C++17 'if constexpr' for compile-time type-specific code generation.
        if constexpr (std::is_same_v<T, c10::Half>) {
            // For half precision, convert to float for computation using `fabsf` (float absolute)
            // and then cast back to c10::Half for storage.
            out[i] = static_cast<c10::Half>(fabsf(static_cast<float>(in[i])));
        } else if constexpr (std::is_same_v<T, float>) {
            // For float precision, use `fabsf` (float absolute) directly.
            out[i] = fabsf(in[i]);
        } else if constexpr (std::is_same_v<T, double>) {
            // For double precision, use `fabs` (double absolute) directly.
            out[i] = fabs(in[i]);
        }
        // No 'else' branch needed here. The host-side dispatch (launch_abs)
        // ensures that only supported types (Float32, Float64, Half) are passed to this kernel.
        // Any unsupported type would have been caught by TORCH_CHECK.
    }
}

// ============ HOST CODE ============

// C++ wrapper function for the absolute value operation.
// This function performs necessary checks, prepares tensors, and launches the CUDA kernel.
torch::Tensor launch_abs(torch::Tensor in_tensor) {
    // --- Input Tensor Checks ---
    TORCH_CHECK(in_tensor.is_cuda(), "Input tensor must be CUDA.");
    TORCH_CHECK(in_tensor.is_contiguous(), "Input tensor must be contiguous. Call .contiguous() on the tensor.");
    TORCH_CHECK(in_tensor.numel() > 0, "Input tensor must not be empty.");

    // Create an output tensor with the same shape, data type, and device as the input.
    auto out_tensor = torch::empty_like(in_tensor);

    // Get the total number of elements for kernel launch configuration.
    int64_t n_elements = in_tensor.numel();

    // --- CUDA Kernel Launch Configuration ---
    // Define a standard number of threads per block.
    const int threads_per_block = 256;
    // Calculate the number of blocks needed to cover all elements.
    // This calculation ensures that enough blocks are launched to process all data.
    const int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // --- Type Dispatch ---
    // Dispatch to the correct templated kernel instance based on the input tensor's scalar type.
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
        // If an unsupported scalar type is encountered, raise a PyTorch-specific error.
        TORCH_CHECK(false, "Unsupported scalar type for abs_kernel: ", in_tensor.scalar_type());
    }

    // --- CUDA Error Checking ---
    // Check for any CUDA errors that might have occurred during the kernel launch.
    // This is crucial for debugging and robust error handling.
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