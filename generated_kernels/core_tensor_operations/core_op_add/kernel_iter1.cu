// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <c10/util/Half.h> // Required for c10::Half type

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise addition
template <typename T>
__global__ void add_kernel(const T* A, const T* B, T* C, int64_t num_elements) {
    // Calculate global index using grid-stride loop
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < num_elements; i += stride) {
        // For half precision, perform computation in float
        // std::is_same is a compile-time check, making this efficient.
        if (std::is_same<T, c10::Half>::value) {
            float val_A = static_cast<float>(A[i]);
            float val_B = static_cast<float>(B[i]);
            C[i] = static_cast<T>(val_A + val_B);
        } else {
            C[i] = A[i] + B[i];
        }
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for element-wise addition
torch::Tensor launch_add(torch::Tensor A, torch::Tensor B) {
    // --- Tensor Checks ---
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be on CUDA.");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be on CUDA.");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Input tensors A and B must have the same data type.");
    TORCH_CHECK(A.sizes() == B.sizes(), "Input tensors A and B must have the same dimensions.");
    TORCH_CHECK(A.numel() > 0, "Input tensor A must not be empty.");

    // Ensure tensors are contiguous for efficient memory access
    A = A.contiguous();
    B = B.contiguous();

    // Create output tensor with the same options and shape as A
    auto C = torch::empty_like(A);

    // Get the total number of elements
    int64_t num_elements = A.numel();

    // --- CUDA Kernel Launch Configuration ---
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    // Cap num_blocks if necessary to avoid exceeding grid limits, though a grid-stride loop makes this less critical.
    // A common cap is 65535, but for large tensors, the computed num_blocks could exceed this if not careful.
    // For simplicity, we'll let the calculation stand, assuming PyTorch will handle large allocations.

    // --- Type Dispatch and Kernel Launch ---
    if (A.scalar_type() == torch::kFloat32) {
        add_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), num_elements
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        add_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), num_elements
        );
    } else if (A.scalar_type() == torch::kHalf) {
        add_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), C.data_ptr<c10::Half>(), num_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type for element-wise addition.");
    }

    // --- CUDA Error Checking ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return C;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_add", &launch_add, "Element-wise addition of two tensors (CUDA)");
}
// [END kernel.cu]