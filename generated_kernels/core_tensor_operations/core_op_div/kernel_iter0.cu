// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string> // Standard library string, potentially useful but not strictly for this op.

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise division: C = A / B
// Handles float, double, and c10::Half types.
template <typename T>
__global__ void div_kernel(const T* A, const T* B, T* C, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Use if constexpr for C++17 compile-time conditional compilation.
        // This allows type-specific logic while keeping the kernel templated.
        if constexpr (std::is_same_v<T, c10::Half>) {
            // For half-precision, convert to float for division and convert back.
            // This ensures higher precision for the intermediate computation.
            C[i] = static_cast<T>(static_cast<float>(A[i]) / static_cast<float>(B[i]));
        } else {
            // For float and double, perform division directly in their native precision.
            C[i] = A[i] / B[i];
        }
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for element-wise division of two tensors on CUDA.
// This function performs necessary checks, allocates output memory,
// configures kernel launch parameters, and dispatches to the correct
// templated kernel based on tensor data type.
torch::Tensor launch_div(torch::Tensor A, torch::Tensor B) {
    // 1. Input Tensor Validation
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor.");
    TORCH_CHECK(A.sizes() == B.sizes(), "Input tensor A and B must have the same sizes for element-wise division.");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Input tensor A and B must have the same scalar type.");

    // Ensure tensors are contiguous for efficient memory access in the kernel.
    // If they are not contiguous, .contiguous() will return a new contiguous tensor.
    A = A.contiguous();
    B = B.contiguous();

    // 2. Output Tensor Allocation
    // Create an output tensor 'C' with the same shape, dtype, and device as 'A'.
    auto C = torch::empty_like(A);

    // 3. Kernel Launch Configuration
    int64_t n_elements = A.numel(); // Total number of elements in the tensor.

    // Handle empty tensors case
    if (n_elements == 0) {
        return C; // Return an empty tensor if there are no elements to process.
    }

    // Define standard CUDA kernel launch parameters.
    // 256 threads per block is a common and efficient choice.
    const int64_t threads_per_block = 256;
    // Calculate the number of blocks needed to cover all elements.
    const int64_t num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // 4. Type Dispatch and Kernel Launch
    // Dispatch to the appropriate templated kernel based on the scalar type of the input tensors.
    if (A.scalar_type() == torch::kFloat32) {
        div_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n_elements
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        div_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), n_elements
        );
    } else if (A.scalar_type() == torch::kHalf) {
        // Use c10::Half for PyTorch's half-precision floating-point type.
        div_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), C.data_ptr<c10::Half>(), n_elements
        );
    } else {
        // If an unsupported type is encountered, raise a PyTorch error.
        TORCH_CHECK(false, "Unsupported scalar type for element-wise division: ", A.scalar_type());
    }

    // 5. CUDA Error Checking
    // After launching the kernel, check for any asynchronous CUDA errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error in launch_div: ", cudaGetErrorString(err));
    }

    return C; // Return the resulting tensor.
}

// ============ PYBIND11 MODULE DEFINITION ============
// This macro defines the entry point for the PyTorch C++ extension.
// TORCH_EXTENSION_NAME is a macro defined by PyTorch's build system,
// typically resolving to the module name specified in setup.py.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Register the 'launch_div' C++ function as 'launch' in the Python module.
    // The third argument is a docstring visible in Python.
    m.def("launch", &launch_div, "Element-wise division of two CUDA tensors (CUDA)");
}
// [END kernel.cu]