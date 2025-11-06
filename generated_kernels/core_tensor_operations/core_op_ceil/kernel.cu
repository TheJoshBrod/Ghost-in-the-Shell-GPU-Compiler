// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string> // Included for general compliance, not strictly needed for this op
#include <cmath>  // For ceilf and ceil
#include <type_traits> // For std::is_same_v

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise ceiling operation
template <typename T>
__global__ void ceil_kernel(const T* A, T* C, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (; idx < n_elements; idx += stride) {
        if constexpr (std::is_same_v<T, double>) {
            // Use double-precision ceil for double type
            C[idx] = static_cast<T>(ceil(static_cast<double>(A[idx])));
        } else {
            // Use single-precision ceil for float and c10::Half types
            C[idx] = static_cast<T>(ceilf(static_cast<float>(A[idx])));
        }
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function to launch the ceiling kernel
torch::Tensor launch_ceil(torch::Tensor A) {
    // --- Input Tensor Checks ---
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.is_contiguous(), "Input tensor A must be contiguous.");

    // --- Output Tensor Creation ---
    // Create an output tensor C with the same shape and options (device, dtype) as A
    auto C = torch::empty_like(A);

    // --- Dimension and Grid/Block Calculation ---
    int64_t n_elements = A.numel(); // Total number of elements in the tensor

    // Standard CUDA launch configuration: 256 threads per block
    int threads_per_block = 256;
    // Calculate number of blocks needed to cover all elements
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // --- Type Dispatch and Kernel Launch ---
    // Manual type dispatch based on the scalar type of tensor A
    if (A.scalar_type() == torch::kFloat32) {
        ceil_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(),
            C.data_ptr<float>(),
            n_elements
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        ceil_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(),
            C.data_ptr<double>(),
            n_elements
        );
    } else if (A.scalar_type() == torch::kHalf) {
        ceil_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(),
            C.data_ptr<c10::Half>(),
            n_elements
        );
    } else {
        // If an unsupported type is encountered, raise an error
        TORCH_CHECK(false, "Unsupported data type for ceil operation: ", A.scalar_type());
    }

    // --- CUDA Error Checking ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return C; // Return the resulting tensor
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Bind the C++ wrapper function to Python, accessible via "launch"
    m.def("launch", &launch_ceil, "Performs element-wise ceiling operation on a CUDA tensor.");
}
// [END kernel.cu]