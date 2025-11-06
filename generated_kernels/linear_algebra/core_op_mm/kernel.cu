// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string> // Required for std::string if used in wrapper, though not directly for mm.

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for matrix multiplication
// A: M x K, B: K x N, C: M x N
template <typename T>
__global__ void mm_kernel(const T* A, const T* B, T* C, int64_t M, int64_t K, int64_t N) {
    // Each thread computes one element of C.
    // Use a grid-stride loop to ensure all elements are covered, especially for larger matrices.
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t total_elements = M * N;

    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t row = i / N; // Row in C (and A)
        int64_t col = i % N; // Column in C (and B)
        
        // Accumulate sum for C[row, col]
        float sum = 0.0f; // Use float for accumulation to maintain precision
        for (int64_t k = 0; k < K; k++) {
            // A[row, k] * B[k, col]
            sum += static_cast<float>(A[row * K + k]) * static_cast<float>(B[k * N + col]);
        }
        C[i] = static_cast<T>(sum); // Store result, casting back to the original type
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// C++ wrapper function to launch the CUDA kernel
// This function takes two PyTorch tensors A and B, performs checks,
// allocates output tensor C, and launches the mm_kernel.
torch::Tensor launch_mm(torch::Tensor A, torch::Tensor B) {
    // --- Tensor Checks ---
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor.");
    TORCH_CHECK(A.dim() == 2, "Input tensor A must be 2-dimensional.");
    TORCH_CHECK(B.dim() == 2, "Input tensor B must be 2-dimensional.");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Input tensors A and B must have the same data type.");
    TORCH_CHECK(A.size(1) == B.size(0), 
                "Dimension mismatch: A's second dimension (K) must match B's first dimension (K). "
                "Got A.size(1) = ", A.size(1), " and B.size(0) = ", B.size(0));

    // Ensure tensors are contiguous for efficient memory access on the device
    // (This also handles cases where input might be non-contiguous, creating a copy if needed)
    A = A.contiguous();
    B = B.contiguous();

    // Get dimensions
    int64_t M = A.size(0); // Rows of A, rows of C
    int64_t K = A.size(1); // Cols of A, rows of B
    int64_t N = B.size(1); // Cols of B, cols of C

    // Create output tensor C with the same options (device, dtype) as A
    auto C = torch::empty({M, N}, A.options());

    // --- Kernel Launch Configuration ---
    // Standard block size for CUDA kernels
    int threads_per_block = 256; 
    // Calculate total elements to process
    int64_t total_elements = M * N;
    // Calculate number of blocks needed to cover all elements
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Clamp number of blocks to avoid excessive launch overhead if total_elements is very small
    // (Optional, but good practice. For simplicity, we can sometimes omit for small kernels.)
    // num_blocks = std::min(num_blocks, 65535); // A common limit for grid dimensions

    // --- Type Dispatch and Kernel Launch ---
    // Use if-else statements for manual type dispatch
    if (A.scalar_type() == torch::kFloat32) {
        mm_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(), // Pointer to A's data
            B.data_ptr<float>(), // Pointer to B's data
            C.data_ptr<float>(), // Pointer to C's data
            M, K, N             // Dimensions
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        mm_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(), 
            B.data_ptr<double>(), 
            C.data_ptr<double>(), 
            M, K, N
        );
    } else if (A.scalar_type() == torch::kHalf) {
        // c10::Half is PyTorch's representation for half-precision floating point
        mm_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(), 
            B.data_ptr<c10::Half>(), 
            C.data_ptr<c10::Half>(), 
            M, K, N
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for matrix multiplication: ", A.scalar_type());
    }

    // --- CUDA Error Checking ---
    // It's crucial to check for CUDA errors after a kernel launch.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error in mm_kernel: ", cudaGetErrorString(err));
    }

    return C;
}

// ============ PYBIND11 MODULE DEFINITION ============

// Defines the Pybind11 module.
// TORCH_EXTENSION_NAME is a macro defined by PyTorch's build system,
// typically resolving to the name of your extension (e.g., "my_extension").
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Binds the C++ function 'launch_mm' to a Python function named 'launch'.
    // The third argument is a docstring that will appear in Python help().
    m.def("launch", &launch_mm, "Custom matrix multiplication (mm) CUDA kernel.");
}
// [END kernel.cu]