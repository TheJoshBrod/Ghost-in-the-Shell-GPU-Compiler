// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Half.h> // Required for c10::Half type

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise multiplication (C = A * B)
template <typename T>
__global__ void mul_kernel(const T* A, const T* B, T* C, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Handle half precision by casting to float for computation
        if constexpr (std::is_same<T, c10::Half>::value) {
            float val_a = static_cast<float>(A[i]);
            float val_b = static_cast<float>(B[i]);
            C[i] = static_cast<T>(val_a * val_b);
        } else {
            C[i] = A[i] * B[i];
        }
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// C++ wrapper function to launch the CUDA kernel
torch::Tensor launch_mul(torch::Tensor A, torch::Tensor B) {
    // Input checks for tensors
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor.");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Input tensors must have the same data type.");
    TORCH_CHECK(A.sizes() == B.sizes(), "Input tensors must have the same shape for element-wise multiplication.");

    // Ensure tensors are contiguous for efficient memory access
    A = A.contiguous();
    B = B.contiguous();

    // Get the total number of elements
    int64_t n_elements = A.numel();

    // Create an output tensor with the same shape and options as input A
    auto C = torch::empty_like(A);

    // Define CUDA launch configuration
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // Manual type dispatch based on scalar type of the input tensors
    if (A.scalar_type() == torch::kFloat32) {
        mul_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n_elements
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        mul_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), n_elements
        );
    } else if (A.scalar_type() == torch::kHalf) {
        mul_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), C.data_ptr<c10::Half>(), n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type for element-wise multiplication. Supported types are float32, float64, and half.");
    }

    // Check for any CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return C;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_mul, "Element-wise multiplication (mul) CUDA kernel.");
}
// [END kernel.cu]