// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string> // Included as per the general template, though not strictly used here.
#include <algorithm> // Required for std::max

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise maximum operation
// C[i] = max(A[i], B[i]) for all elements i
template <typename T>
__global__ void maximum_kernel(const T* A, const T* B, T* C, int64_t n_elements) {
    // Calculate global index using grid-stride loop
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Perform element-wise maximum.
        // For c10::Half, std::max works directly due to operator overloads.
        C[i] = std::max(A[i], B[i]);
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// C++ wrapper function to launch the CUDA kernel for element-wise maximum
torch::Tensor launch_maximum(torch::Tensor A, torch::Tensor B) {
    // --- Input Tensor Checks ---
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor.");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Input tensors must have the same scalar type.");
    // For a simple element-wise maximum, assume identical shapes.
    // More complex broadcasting would require additional logic.
    TORCH_CHECK(A.sizes().vec() == B.sizes().vec(), "Input tensors must have the same shape for element-wise maximum.");

    // Ensure input tensors are contiguous to simplify kernel indexing
    A = A.contiguous();
    B = B.contiguous();

    // Get the total number of elements in the tensors
    int64_t n_elements = A.numel();

    // Create the output tensor C with the same properties (shape, type, device) as A
    auto C = torch::empty_like(A);

    // --- CUDA Kernel Launch Configuration ---
    // Standard block size of 256 threads
    int threads_per_block = 256;
    // Calculate the number of blocks needed to cover all elements
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // --- Type Dispatch ---
    // Launch the appropriate templated kernel based on the scalar type of the tensors
    if (A.scalar_type() == torch::kFloat32) {
        maximum_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n_elements
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        maximum_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), n_elements
        );
    } else if (A.scalar_type() == torch::kHalf) {
        // c10::Half is PyTorch's type for half-precision, which works with std::max
        maximum_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), C.data_ptr<c10::Half>(), n_elements
        );
    } else {
        // Throw an error for unsupported data types
        TORCH_CHECK(false, "Unsupported scalar type for maximum operation: ", A.scalar_type());
    }

    // --- CUDA Error Checking ---
    // Check for any CUDA errors that occurred during or after the kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return C;
}

// Pybind11 module definition
// This creates the Python binding for the custom operator.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Define the "launch" function in the Python module,
    // which calls our C++ wrapper `launch_maximum`.
    m.def("launch", &launch_maximum, "CUDA Element-wise Maximum operation");
}
// [END kernel.cu]