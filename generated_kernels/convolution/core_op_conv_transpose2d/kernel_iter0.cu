// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for elementwise addition
template <typename T>
__global__ void add_kernel(const T* A, const T* B, T* C, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n_elements; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function
torch::Tensor launch_add(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    
    TORCH_CHECK(A.sizes() == B.sizes(), "A and B must have the same shape");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must have the same data type");
    
    A = A.contiguous();
    B = B.contiguous();
    
    auto C = torch::empty_like(A);

    int64_t n_elements = A.numel();
    int threads = 256;
    int blocks = (n_elements + threads - 1) / threads;

    if (A.scalar_type() == torch::kFloat32) {
        add_kernel<float><<<blocks, threads>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n_elements
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        add_kernel<double><<<blocks, threads>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), n_elements
        );
    } else if (A.scalar_type() == torch::kHalf) {
        add_kernel<c10::Half><<<blocks, threads>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), C.data_ptr<c10::Half>(), n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return C;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_add", &launch_add, "Elementwise addition kernel");
}
// [END kernel.cu]