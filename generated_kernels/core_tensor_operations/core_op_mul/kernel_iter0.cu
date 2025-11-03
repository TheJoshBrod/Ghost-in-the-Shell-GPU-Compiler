// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string> // Required for std::string arguments if any, good practice to include

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise multiplication
template <typename T>
__global__ void mul_kernel(const T* A, const T* B, T* C, int64_t n_elements) {
    // Calculate global index using grid-stride loop
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Perform multiplication. For half precision, cast to float for computation.
        float val_a = static_cast<float>(A[i]);
        float val_b = static_cast<float>(B[i]);
        C[i] = static_cast<T>(val_a * val_b);
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// C++ wrapper function to launch the CUDA kernel
torch::Tensor launch_mul(torch::Tensor A, torch::Tensor B) {
    // 1. Input Validation and Checks
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor.");
    
    TORCH_CHECK(A.dim() == 2, "Input tensor A must be 2-dimensional.");
    TORCH_CHECK(B.dim() == 2, "Input tensor B must be 2-dimensional.");

    TORCH_CHECK(A.sizes().vec() == B.sizes().vec(), 
                "Input tensors A and B must have the same shape, but got A: ", A.sizes(), " and B: ", B.sizes());
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), 
                "Input tensors A and B must have the same data type, but got A: ", A.scalar_type(), " and B: ", B.scalar_type());

    // Ensure tensors are contiguous for simple pointer arithmetic in kernel
    A = A.contiguous();
    B = B.contiguous();

    // 2. Determine output tensor properties
    int64_t M = A.size(0);
    int64_t N = A.size(1);
    int64_t n_elements = M * N;

    auto C = torch::empty_like(A); // Create output tensor with same shape and options as A

    // 3. Configure kernel launch parameters
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // 4. Dispatch to appropriate kernel based on data type
    if (A.scalar_type() == torch::kFloat32) {
        mul_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n_elements
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        mul_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), n_elements
        );
    } else if (A.scalar_type() == torch::kHalf) {
        // c10::Half is PyTorch's type for half-precision floating point
        mul_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), C.data_ptr<c10::Half>(), n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type for element-wise multiplication kernel. Supported types are float32, float64, and float16.");
    }

    // 5. Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch error for mul_kernel: ", cudaGetErrorString(err));
    }

    // 6. Return the result tensor
    return C;
}

// PYBIND11_MODULE for Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_mul, "Element-wise multiplication of two tensors (A * B).");
}
// [END kernel.cu]