// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for batched matrix multiplication C = A @ B
// A: B x M x K
// B: B x K x N
// C: B x M x N
template <typename T>
__global__ void bmm_kernel(
    const T* A, 
    const T* B, 
    T* C, 
    int64_t B_dim, 
    int64_t M_dim, 
    int64_t K_dim, 
    int64_t N_dim
) {
    // Calculate total elements in the output tensor C
    int64_t total_elements = B_dim * M_dim * N_dim;

    // Use a grid-stride loop to ensure all elements are processed
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        // Deconstruct the linear index 'i' into (batch, row, col)
        int64_t elements_per_batch = M_dim * N_dim;
        int64_t batch = i / elements_per_batch;
        int64_t flat_mn_idx = i % elements_per_batch;
        int64_t row = flat_mn_idx / N_dim;
        int64_t col = flat_mn_idx % N_dim;
        
        float sum = 0.0f;
        // Perform dot product for C[batch, row, col]
        for (int64_t k = 0; k < K_dim; k++) {
            // A access: A[batch * M_dim * K_dim + row * K_dim + k]
            // B access: B[batch * K_dim * N_dim + k * N_dim + col]
            sum += static_cast<float>(A[batch * M_dim * K_dim + row * K_dim + k]) *
                   static_cast<float>(B[batch * K_dim * N_dim + k * N_dim + col]);
        }
        // Store the result in C[batch, row, col]
        C[i] = static_cast<T>(sum);
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for batched matrix multiplication
torch::Tensor launch_bmm(torch::Tensor A, torch::Tensor B) {
    // Input Tensor Checks
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor.");
    TORCH_CHECK(A.dim() == 3, "Input tensor A must be 3-dimensional (B, M, K).");
    TORCH_CHECK(B.dim() == 3, "Input tensor B must be 3-dimensional (B, K, N).");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Input tensors A and B must have the same data type.");

    // Dimension Checks
    // A: B x M x K, B: B x K x N
    // Output C: B x M x N
    int64_t B_dim = A.size(0);
    int64_t M_dim = A.size(1);
    int64_t K_dim_A = A.size(2);
    int64_t B_dim_B = B.size(0);
    int64_t K_dim_B = B.size(1);
    int64_t N_dim = B.size(2);

    TORCH_CHECK(B_dim == B_dim_B, "Batch dimensions must match for A and B.");
    TORCH_CHECK(K_dim_A == K_dim_B, "Inner dimensions (K) must match for A and B.");

    // Ensure tensors are contiguous for simpler pointer arithmetic in kernel
    A = A.contiguous();
    B = B.contiguous();

    // Create the output tensor C with the same options (device, dtype) as A
    auto C = torch::empty({B_dim, M_dim, N_dim}, A.options());

    // Calculate launch configuration
    int threads_per_block = 256;
    int64_t total_elements = B_dim * M_dim * N_dim;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Type dispatch to call the templated kernel
    if (A.scalar_type() == torch::kFloat32) {
        bmm_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(), 
            B.data_ptr<float>(), 
            C.data_ptr<float>(), 
            B_dim, M_dim, K_dim_A, N_dim
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        bmm_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(), 
            B.data_ptr<double>(), 
            C.data_ptr<double>(), 
            B_dim, M_dim, K_dim_A, N_dim
        );
    } else if (A.scalar_type() == torch::kHalf) {
        bmm_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(), 
            B.data_ptr<c10::Half>(), 
            C.data_ptr<c10::Half>(), 
            B_dim, M_dim, K_dim_A, N_dim
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for bmm_kernel: ", A.scalar_type());
    }

    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error in launch_bmm: ", cudaGetErrorString(err));
    }

    return C;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_bmm, "Batched Matrix Multiplication CUDA kernel (BMM)");
}
// [END kernel.cu]