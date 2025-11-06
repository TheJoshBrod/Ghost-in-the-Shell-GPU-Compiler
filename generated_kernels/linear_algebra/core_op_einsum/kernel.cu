// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <c10/util/Half.h>      // Required for c10::Half type
#include <ATen/ATen.h>          // For AT_DISPATCH_FLOATING_TYPES_AND_HALF
#include <ATen/Dispatch.h>      // For AT_DISPATCH_FLOATING_TYPES_AND_HALF
#include <type_traits>          // For std::conditional and std::is_same

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
    // Determine the accumulation type for higher precision:
    // float for c10::Half, otherwise use the input type T (float or double).
    using acc_t = typename std::conditional<std::is_same<T, c10::Half>::value, float, T>::type;

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
        
        acc_t sum = static_cast<acc_t>(0.0);
        // Perform dot product for C[batch, row, col]
        for (int64_t k = 0; k < K_dim; k++) {
            // A access: A[batch * M_dim * K_dim + row * K_dim + k]
            // B access: B[batch * K_dim * N_dim + k * N_dim + col]
            sum += static_cast<acc_t>(A[batch * M_dim * K_dim + row * K_dim + k]) *
                   static_cast<acc_t>(B[batch * K_dim * N_dim + k * N_dim + col]);
        }
        // Store the result in C[batch, row, col]
        C[i] = static_cast<T>(sum);
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for batched matrix multiplication
// The 'equation' string is accepted to match calling signature but not used by this BMM kernel.
torch::Tensor launch_bmm(const std::string& equation, torch::Tensor A_in, torch::Tensor B_in) {
    // Input Tensor Checks
    TORCH_CHECK(A_in.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(B_in.is_cuda(), "Input tensor B must be a CUDA tensor.");
    TORCH_CHECK(A_in.scalar_type() == B_in.scalar_type(), "Input tensors A and B must have the same data type.");

    // Allow 2D or 3D inputs, consistent with torch.bmm behavior
    TORCH_CHECK(A_in.dim() == 2 || A_in.dim() == 3, "Input tensor A must be 2-dimensional (M, K) or 3-dimensional (B, M, K).");
    TORCH_CHECK(B_in.dim() == 2 || B_in.dim() == 3, "Input tensor B must be 2-dimensional (K, N) or 3-dimensional (B, K, N).");

    torch::Tensor A = A_in;
    torch::Tensor B = B_in;

    // Handle 2D inputs by unsqueezing to add a batch dimension of 1
    if (A.dim() == 2) {
        A = A.unsqueeze(0); // (M, K) -> (1, M, K)
    }
    if (B.dim() == 2) {
        B = B.unsqueeze(0); // (K, N) -> (1, K, N)
    }

    // Now both A and B are guaranteed to be 3D:
    // A: B_A x M x K_A
    // B: B_B x K_B x N
    int64_t B_dim_A = A.size(0);
    int64_t M_dim = A.size(1);
    int64_t K_dim_A = A.size(2);
    int64_t B_dim_B = B.size(0);
    int64_t K_dim_B = B.size(1);
    int64_t N_dim = B.size(2);

    // Dimension Checks
    TORCH_CHECK(K_dim_A == K_dim_B, "Inner dimensions (K) must match for A (", K_dim_A, ") and B (", K_dim_B, ").");

    int64_t output_B_dim; // Final batch dimension for output C

    // Determine the output batch dimension and handle broadcasting rules (if one batch dim is 1)
    if (B_dim_A == B_dim_B) {
        output_B_dim = B_dim_A;
    } else if (B_dim_A == 1) {
        output_B_dim = B_dim_B;
        A = A.expand({output_B_dim, M_dim, K_dim_A}); // Broadcast A to match B's batch dim
    } else if (B_dim_B == 1) {
        output_B_dim = B_dim_A;
        B = B.expand({output_B_dim, K_dim_B, N_dim}); // Broadcast B to match A's batch dim
    } else {
        TORCH_CHECK(false, "Batch dimensions are incompatible: A batch dimension (", B_dim_A, ") and B batch dimension (", B_dim_B, ") must either match or one of them must be 1.");
    }
    
    // Ensure tensors are contiguous AFTER any unsqueeze/expand operations for simpler pointer arithmetic in kernel
    A = A.contiguous();
    B = B.contiguous();

    // Create the output tensor C with the determined dimensions
    auto C = torch::empty({output_B_dim, M_dim, N_dim}, A.options());

    // Calculate launch configuration
    int threads_per_block = 256;
    int64_t total_elements = output_B_dim * M_dim * N_dim;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    if (num_blocks == 0 && total_elements > 0) num_blocks = 1; // Ensure at least one block if there are elements

    // Type dispatch to call the templated kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "bmm_kernel", [&] {
        bmm_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(), 
            B.data_ptr<scalar_t>(), 
            C.data_ptr<scalar_t>(), 
            output_B_dim, M_dim, K_dim_A, N_dim
        );
    });

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