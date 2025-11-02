// [START kernel.cu]
#include <c10/util/Half.h>
#include <cuda_runtime.h>
#include <ATen/Dispatch.h>
#include <torch/extension.h>

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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "add_kernel", ([&] {
        add_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(), n_elements
        );
    }));

    cudaError_t err = cudaDeviceSynchronize();
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