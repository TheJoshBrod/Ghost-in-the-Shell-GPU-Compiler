// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string> // Included as per general requirements, though not directly used in this specific op

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for matrix multiplication (C = A @ B)
// A: M x K, B: K x N, C: M x N
template <typename T>
__global__ void mm_kernel(const T* A, const T* B, T* C, int64_t M, int64_t K, int64_t N) {
    // Each thread computes one element of C
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t total_elements = M * N;

    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t row = i / N; // Current row in C and A
        int64_t col = i % N; // Current column in C and B
        
        // Accumulate sum for C[row][col]
        float sum = 0.0f; // Use float for accumulation to maintain precision
        for (int64_t k = 0; k < K; k++) {
            // A[row][k] * B[k][col]
            sum += static_cast<float>(A[row * K + k]) * static_cast<float>(B[k * N + col]);
        }
        C[i] = static_cast<T>(sum); // Store result, converting back to original type
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for the matrix multiplication operation
torch::Tensor launch_mm(torch::Tensor A, torch::Tensor B) {
    // --- Input Tensor Checks ---
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor.");
    TORCH_CHECK(A.dim() == 2, "Input tensor A must be 2-dimensional (matrix).");
    TORCH_CHECK(B.dim() == 2, "Input tensor B must be 2-dimensional (matrix).");
    TORCH_CHECK(A.size(1) == B.size(0), 
                "Dimension mismatch: A's columns must match B's rows for matrix multiplication. "
                "A has size (", A.size(0), ", ", A.size(1), ") and B has size (", B.size(0), ", ", B.size(1), ").");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), 
                "Type mismatch: Input tensors A and B must have the same data type. "
                "A is of type ", A.scalar_type(), " and B is of type ", B.scalar_type(), ".");

    // Ensure tensors are contiguous for proper memory access in kernel
    A = A.contiguous();
    B = B.contiguous();

    // Get dimensions
    int64_t M = A.size(0); // Rows of A
    int64_t K = A.size(1); // Columns of A, Rows of B
    int64_t N = B.size(1); // Columns of B

    // Create output tensor C, initialized on the same device with same options as A
    auto C = torch::empty({M, N}, A.options());

    // --- CUDA Kernel Launch Configuration ---
    int threads_per_block = 256;
    int64_t total_elements = M * N;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // --- Type Dispatch and Kernel Launch ---
    if (A.scalar_type() == torch::kFloat32) {
        mm_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        mm_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), M, K, N
        );
    } else if (A.scalar_type() == torch::kHalf) {
        // For half precision, use c10::Half
        mm_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), C.data_ptr<c10::Half>(), M, K, N
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for matrix multiplication: ", A.scalar_type());
    }

    // --- Check for CUDA errors after kernel launch ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return C;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_mm, "Custom Matrix Multiplication (CUDA)");
}
// [END kernel.cu]