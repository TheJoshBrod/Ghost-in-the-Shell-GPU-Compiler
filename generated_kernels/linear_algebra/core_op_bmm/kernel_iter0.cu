// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for batched matrix multiplication (BMM)
// A: B_dim x M x K
// B: B_dim x K x N
// C: B_dim x M x N
template <typename T>
__global__ void bmm_kernel(const T* A, const T* B, T* C, int64_t B_dim, int64_t M, int64_t K, int64_t N) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t total_elements_per_batch = M * N;
    int64_t total_elements = B_dim * total_elements_per_batch;

    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t batch_idx = i / total_elements_per_batch;
        int64_t rem_in_batch = i % total_elements_per_batch;
        int64_t row = rem_in_batch / N;
        int64_t col = rem_in_batch % N;
        
        // Calculate offsets for A, B, C
        int64_t A_batch_offset = batch_idx * M * K;
        int64_t B_batch_offset = batch_idx * K * N;
        int64_t C_batch_offset = batch_idx * M * N;

        float sum = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            sum += static_cast<float>(A[A_batch_offset + row * K + k]) * 
                   static_cast<float>(B[B_batch_offset + k * N + col]);
        }
        C[C_batch_offset + row * N + col] = static_cast<T>(sum);
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for batched matrix multiplication
torch::Tensor launch_bmm(torch::Tensor A, torch::Tensor B) {
    // --- Input Checks ---
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be on CUDA device.");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be on CUDA device.");

    TORCH_CHECK(A.dim() == 3, "Input tensor A must be 3-dimensional (B, M, K).");
    TORCH_CHECK(B.dim() == 3, "Input tensor B must be 3-dimensional (B, K, N).");

    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Input tensors A and B must have the same data type.");

    // Extract dimensions
    int64_t B_dim = A.size(0); // Batch size
    int64_t M = A.size(1);
    int64_t K = A.size(2);
    int64_t N = B.size(2);

    // Dimension checks for BMM
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes of A and B must match.");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) of A and B must match.");

    // Ensure tensors are contiguous for efficient memory access
    A = A.contiguous();
    B = B.contiguous();

    // --- Output Tensor Allocation ---
    auto C = torch::empty({B_dim, M, N}, A.options());

    // --- Kernel Launch Configuration ---
    // Standard block size
    int threads_per_block = 256;
    // Total number of elements in the output tensor
    int64_t total_output_elements = B_dim * M * N;
    // Calculate grid size to cover all elements
    int num_blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;

    // --- Type Dispatch and Kernel Launch ---
    if (A.scalar_type() == torch::kFloat32) {
        bmm_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), B_dim, M, K, N
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        bmm_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), B_dim, M, K, N
        );
    } else if (A.scalar_type() == torch::kHalf) {
        bmm_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), C.data_ptr<c10::Half>(), B_dim, M, K, N
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for batched matrix multiplication.");
    }

    // --- CUDA Error Checking ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return C;
}

// --- Pybind11 Module Definition ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_bmm, "Batched Matrix Multiplication (BMM) CUDA kernel");
}
// [END kernel.cu]