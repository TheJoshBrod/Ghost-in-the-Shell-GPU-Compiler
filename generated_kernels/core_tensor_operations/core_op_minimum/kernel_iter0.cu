// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string> // For std::string, though not used in this specific op
#include <algorithm> // For std::fmin, or custom min for generic types

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise minimum operation
// C[i] = min(A[i], B[i]) for all i
template <typename T>
__global__ void minimum_kernel(const T* A, const T* B, T* C, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Special handling for __half (FP16)
        if constexpr (std::is_same_v<T, __half>) {
            // Convert to float for computation to avoid precision issues
            // and use fminf for float.
            float val_a = __half2float(A[i]);
            float val_b = __half2float(B[i]);
            C[i] = __float2half(fminf(val_a, val_b));
        } else {
            // For float and double, direct comparison works or use std::min
            // Direct comparison is often optimized better by the compiler
            C[i] = A[i] < B[i] ? A[i] : B[i];
        }
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// C++ wrapper function to launch the minimum kernel
torch::Tensor launch_minimum(torch::Tensor A, torch::Tensor B) {
    // Input Validation
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be on a CUDA device.");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be on a CUDA device.");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Input tensors must have the same data type.");
    TORCH_CHECK(A.sizes() == B.sizes(), "Input tensors must have the same shape.");

    // Ensure tensors are contiguous for efficient memory access in the kernel
    A = A.contiguous();
    B = B.contiguous();

    // Create the output tensor with the same shape and options as input A
    auto C = torch::empty_like(A);

    // Get the total number of elements
    int64_t n_elements = A.numel();

    // Determine kernel launch configuration
    // Standard practice: 256 threads per block
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // Manual type dispatch to call the templated kernel
    if (A.scalar_type() == torch::kFloat32) {
        minimum_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n_elements
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        minimum_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), n_elements
        );
    } else if (A.scalar_type() == torch::kHalf) {
        minimum_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), C.data_ptr<c10::Half>(), n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type for minimum operation. Only Float32, Float64, and Half are supported.");
    }

    // Check for any CUDA errors that occurred during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return C;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_minimum, "Performs element-wise minimum of two tensors on CUDA.");
}
// [END kernel.cu]