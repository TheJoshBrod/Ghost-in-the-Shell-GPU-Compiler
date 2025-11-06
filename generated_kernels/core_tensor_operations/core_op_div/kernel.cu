// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string> // Required for std::string arguments if any, though not for aten::div

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise division
// C[i] = A[i] / B[i]
template <typename T>
__global__ void div_kernel(const T* A, const T* B, T* C, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Perform division. For floating-point types, division by zero will naturally
        // produce +/-INF or NaN, which is typically the expected behavior for aten::div.
        C[i] = static_cast<T>(static_cast<float>(A[i]) / static_cast<float>(B[i]));
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for element-wise division
torch::Tensor launch_div(torch::Tensor A, torch::Tensor B) {
    // --- Argument Checks ---
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be CUDA tensor.");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be CUDA tensor.");
    TORCH_CHECK(A.dim() == B.dim(), "Input tensors must have the same number of dimensions.");
    TORCH_CHECK(A.sizes().vec() == B.sizes().vec(), "Input tensors must have the same shape.");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Input tensors must have the same scalar type.");

    // Ensure tensors are contiguous for efficient memory access
    A = A.contiguous();
    B = B.contiguous();

    // Get the total number of elements
    int64_t n_elements = A.numel();

    // Allocate output tensor C with the same shape and options as A
    auto C = torch::empty_like(A);

    // --- CUDA Kernel Configuration ---
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;
    // Cap num_blocks to avoid excessive grid dimensions, if n_elements is extremely large
    // A common practical limit is 65535 for older devices, newer devices support much more.
    // However, for typical tensor sizes, this calculation is usually fine.
    if (num_blocks > 65535) { // Example cap, adjust as needed based on target hardware/CUDA version
        num_blocks = 65535;
    }


    // --- Type Dispatch and Kernel Launch ---
    if (A.scalar_type() == torch::kFloat32) {
        div_kernel<float><<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n_elements
        );
    } else if (A.scalar_type() == torch::kFloat64) {
        div_kernel<double><<<num_blocks, threads_per_block>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), C.data_ptr<double>(), n_elements
        );
    } else if (A.scalar_type() == torch::kHalf) {
        div_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            A.data_ptr<c10::Half>(), B.data_ptr<c10::Half>(), C.data_ptr<c10::Half>(), n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for element-wise division.");
    }

    // --- CUDA Error Checking ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return C;
}

// Pybind11 binding for the extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_div, "Element-wise division (CUDA)");
}
// [END kernel.cu]