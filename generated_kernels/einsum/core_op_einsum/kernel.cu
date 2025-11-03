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
// Modified to correctly handle einsum patterns for 2D (ij,jk->ik) and 3D (bik,bkj->bij) inputs,
// which map to the underlying 3D BMM kernel.
// The signature is changed to directly accept two tensors, matching how PyTorch's `einsum`
// might pass arguments for a binary operation, and resolving the `incompatible function arguments` error.
torch::Tensor launch_bmm(std::string equation, torch::Tensor A_orig, torch::Tensor B_orig) {
    // --- Input Validation for Device and Type ---
    TORCH_CHECK(A_orig.is_cuda(), "Input A must be a CUDA tensor.");
    TORCH_CHECK(B_orig.is_cuda(), "Input B must be a CUDA tensor.");
    TORCH_CHECK(A_orig.scalar_type() == B_orig.scalar_type(), "Input tensors must have the same data type.");

    torch::Tensor A_processed, B_processed;
    bool is_2d_input = false; // Flag to indicate if original inputs were 2D

    // --- Dimension Handling based on Einsum Equation ---
    if (equation == "ij,jk->ik") {
        TORCH_CHECK(A_orig.dim() == 2, "Input A for 'ij,jk->ik' must be 2D, but got ", A_orig.dim(), "D.");
        TORCH_CHECK(B_orig.dim() == 2, "Input B for 'ij,jk->ik' must be 2D, but got ", B_orig.dim(), "D.");
        TORCH_CHECK(A_orig.size(1) == B_orig.size(0), "Inner dimensions must match: A.K (", A_orig.size(1), ") == B.K (", B_orig.size(0), ") for 'ij,jk->ik'.");
        
        // Unsqueeze 2D inputs to 3D (batch size 1) for the BMM kernel
        A_processed = A_orig.unsqueeze(0); // Becomes 1xMxK
        B_processed = B_orig.unsqueeze(0); // Becomes 1xKxN
        is_2d_input = true;
    } else if (equation == "bik,bkj->bij") {
        TORCH_CHECK(A_orig.dim() == 3, "Input A for 'bik,bkj->bij' must be 3D, but got ", A_orig.dim(), "D.");
        TORCH_CHECK(B_orig.dim() == 3, "Input B for 'bik,bkj->bij' must be 3D, but got ", B_orig.dim(), "D.");
        TORCH_CHECK(A_orig.size(0) == B_orig.size(0), "Batch dimensions must match for A and B (", A_orig.size(0), " vs ", B_orig.size(0), ").");
        TORCH_CHECK(A_orig.size(2) == B_orig.size(1), "Inner dimensions must match: A.K (", A_orig.size(2), ") == B.K (", B_orig.size(1), ") for 'bik,bkj->bij'.");

        A_processed = A_orig;
        B_processed = B_orig;
    } else {
        TORCH_CHECK(false, "Unsupported einsum equation for custom BMM: '", equation,
                    "'. This custom BMM kernel currently only supports 'ij,jk->ik' or 'bik,bkj->bij' patterns.");
    }

    // --- Ensure Contiguity ---
    // Make copies if not contiguous. This is generally good practice for performance
    // if the kernel expects dense memory access patterns.
    A_processed = A_processed.contiguous();
    B_processed = B_processed.contiguous();

    // --- Extract Dimensions from the (now 3D) tensors for kernel ---
    int64_t B_dim = A_processed.size(0); // Batch size (will be 1 for 2D inputs)
    int64_t M = A_processed.size(1);    // Rows of A_batch and C_batch
    int64_t K = A_processed.size(2);    // Columns of A_batch and Rows of B_batch
    int64_t N = B_processed.size(2);    // Columns of B_batch and C_batch

    // --- Create Output Tensor (will be 3D initially) ---
    auto C_temp = torch::empty({B_dim, M, N}, A_processed.options());

    // --- Kernel Launch Configuration ---
    int threads_per_block = 256; // Standard block size
    int64_t total_output_elements = B_dim * M * N;
    int num_blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    if (num_blocks == 0 && total_output_elements > 0) num_blocks = 1; // Ensure at least one block if there are elements
    if (total_output_elements == 0) num_blocks = 0; // No blocks if output is empty

    // --- Type Dispatch and Kernel Launch ---
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A_processed.scalar_type(), "bmm_kernel", ([&] {
        if (total_output_elements > 0) { // Only launch if there's actual work to do
            bmm_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
                A_processed.data_ptr<scalar_t>(),
                B_processed.data_ptr<scalar_t>(),
                C_temp.data_ptr<scalar_t>(),
                B_dim, M, K, N
            );
        }
    }));

    // --- CUDA Error Checking ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    // --- Squeeze output if inputs were originally 2D ---
    if (is_2d_input) {
        return C_temp.squeeze(0); // Becomes MxN
    } else {
        return C_temp; // Remains BxMxN
    }
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // The Pybind11 definition now matches the updated C++ function signature,
    // which accepts a string and two torch::Tensor objects directly.
    m.def("launch", &launch_bmm, "Batch Matrix Multiplication (BMM) CUDA kernel, compatible with einsum 'ij,jk->ik' or 'bik,bkj->bij' for 2D/3D tensors.");
}
// [END kernel.cu]