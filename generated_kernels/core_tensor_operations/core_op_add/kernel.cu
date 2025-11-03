// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string> // Not strictly needed for this op, but good practice to include

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise addition: C = A + B
template <typename T>
__global__ void add_kernel(const T* A, const T* B, T* C, int64_t n_elements) {
    // Calculate a global index using grid-stride loop
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Perform computation in float for better precision, then cast back to T
        float val_a = static_cast<float>(A[i]);
        float val_b = static_cast<float>(B[i]);
        C[i] = static_cast<T>(val_a + val_b);
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for the element-wise addition kernel
torch::Tensor launch_add(torch::Tensor A, torch::Tensor B) {
    // --- Input Tensor Checks ---
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Input tensors must have the same data type.");
    TORCH_CHECK(A.sizes() == B.sizes(), "Input tensors must have the same shape.");

    // Ensure tensors are contiguous for efficient memory access if not already.
    // For element-wise ops, this is often handled by PyTorch's backend, but explicit
    // contiguity can sometimes simplify kernel design. For add, it's not strictly
    // necessary for correctness but can ensure predictable access patterns.
    A = A.contiguous();
    B = B.contiguous();

    // Determine the total number of elements
    int64_t n_elements = A.numel();

    // Create the output tensor C with the same shape and options as A
    auto C = torch::empty_like(A);

    // --- Kernel Launch Configuration ---
    // Standard CUDA block size
    int threads_per_block = 256;
    // Calculate the number of blocks needed to cover all elements
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // --- Type Dispatch and Kernel Launch ---
    if (A.scalar_type() == torch::kFloat32) {
        add_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n_elements
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        add_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), n_elements
        );
    } else if (A.scalar_type() == torch::kHalf) {
        add_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), C.data_ptr<c10::Half>(), n_elements
        );
    } else {
        // Raise an error for unsupported data types
        TORCH_CHECK(false, "Unsupported scalar type for add operation: ", A.scalar_type());
    }

    // --- CUDA Error Checking ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return C;
}

// --- Pybind11 Module Definition ---
// This macro creates the entry point for the PyTorch C++ extension.
// TORCH_EXTENSION_NAME is defined in setup.py
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Define a Python function named "launch" that calls our C++ wrapper "launch_add"
    m.def("launch", &launch_add, "Element-wise addition CUDA kernel (C = A + B)");
}
// [END kernel.cu]