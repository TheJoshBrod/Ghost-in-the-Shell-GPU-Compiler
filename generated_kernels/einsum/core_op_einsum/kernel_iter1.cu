// [START kernel.cu]
#include <c10/util/Half.h>
#include <cuda_runtime.h>
#include <string>
#include <torch/extension.h>
#include <ATen/Dispatch.h> // Required for AT_DISPATCH macros and scalar_t

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for Batch Matrix Multiplication (BMM)
// A: B_dim x M x K, B: B_dim x K x N, C: B_dim x M x N
template <typename T>
__global__ void bmm_kernel(
    const T* A,
    const T* B,
    T* C,
    int64_t B_dim,
    int64_t M,
    int64_t K,
    int64_t N
) {
    int64_t total_output_elements = B_dim * M * N;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < total_output_elements; i += stride) {
        int64_t batch_idx = i / (M * N);
        int64_t remaining_idx = i % (M * N);
        int64_t row_idx = remaining_idx / N; // M-dimension index for current batch
        int64_t col_idx = remaining_idx % N; // N-dimension index for current batch

        float sum = 0.0f; // Use float for accumulation to prevent precision loss for Half/Float
        // Pointers for current batch's matrices
        const T* A_batch = A + batch_idx * M * K;
        const T* B_batch = B + batch_idx * K * N;

        for (int64_t k_idx = 0; k_idx < K; k_idx++) {
            // A_batch[row_idx][k_idx] * B_batch[k_idx][col_idx]
            sum += static_cast<float>(A_batch[row_idx * K + k_idx]) *
                   static_cast<float>(B_batch[k_idx * N + col_idx]);
        }
        C[i] = static_cast<T>(sum);
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============

// Wrapper function for Batch Matrix Multiplication
// The error message indicated an 'einsum'-like invocation with a string argument.
// We modify the signature to accept this string, while still implementing BMM.
torch::Tensor launch_bmm(std::string equation, torch::Tensor A, torch::Tensor B) {
    // --- Input Validation ---
    // Check if the einsum string matches the expected BMM pattern for 3D tensors.
    TORCH_CHECK(equation == "ij,jk->ik" || equation == "bik,bkj->bij",
                "Unsupported einsum equation for custom BMM: ", equation,
                ". This custom BMM kernel currently only supports 'ij,jk->ik' (for 3D tensors acting as batched) or 'bik,bkj->bij' patterns.");

    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor.");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor.");
    TORCH_CHECK(A.dim() == 3, "Input A must be a 3D tensor (Batch, M, K).");
    TORCH_CHECK(B.dim() == 3, "Input B must be a 3D tensor (Batch, K, N).");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Input tensors must have the same data type.");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch dimensions must match for A and B.");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions must match: A.K == B.K.");

    // --- Ensure Contiguity ---
    // Make copies if not contiguous. This is generally good practice for performance
    // if the kernel expects dense memory access patterns.
    A = A.contiguous();
    B = B.contiguous();

    // --- Extract Dimensions ---
    int64_t B_dim = A.size(0); // Batch size
    int64_t M = A.size(1);    // Rows of A_batch and C_batch
    int64_t K = A.size(2);    // Columns of A_batch and Rows of B_batch
    int64_t N = B.size(2);    // Columns of B_batch and C_batch

    // --- Create Output Tensor ---
    auto C = torch::empty({B_dim, M, N}, A.options());

    // --- Kernel Launch Configuration ---
    int threads_per_block = 256; // Standard block size
    int64_t total_output_elements = B_dim * M * N;
    int num_blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    // Note: num_blocks can be capped to gridDim.x max (e.g., 65535 for older architectures)
    // but modern CUDA generally supports larger grids. 'int' is sufficient for typical sizes.

    // --- Type Dispatch and Kernel Launch ---
    // Using AT_DISPATCH_FLOATING_TYPES_AND_HALF for robust type handling
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "bmm_kernel", ([&] {
        bmm_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            B_dim, M, K, N
        );
    }));

    // --- CUDA Error Checking ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return C;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // The Pybind11 definition must match the updated C++ function signature
    m.def("launch", &launch_bmm, "Batch Matrix Multiplication (BMM) CUDA kernel, compatible with einsum 'ij,jk->ik' or 'bik,bkj->bij' for 3D tensors.");
}
// [END kernel.cu]