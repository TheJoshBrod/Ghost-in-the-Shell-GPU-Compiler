// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise 'where' operation
// out[i] = condition[i] ? x[i] : y[i]
template <typename T>
__global__ void where_kernel(const bool* condition, const T* x, const T* y, T* out, int64_t num_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < num_elements; i += stride) {
        out[i] = condition[i] ? x[i] : y[i];
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for the 'where' operation
torch::Tensor launch_where(torch::Tensor condition, torch::Tensor x, torch::Tensor y) {
    // Input checks
    TORCH_CHECK(condition.is_cuda(), "Condition tensor must be CUDA");
    TORCH_CHECK(x.is_cuda(), "X tensor must be CUDA");
    TORCH_CHECK(y.is_cuda(), "Y tensor must be CUDA");

    TORCH_CHECK(condition.scalar_type() == torch::kBool, "Condition tensor must be of type bool");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "X and Y tensors must have the same scalar type");

    // Determine the output shape by broadcasting all three inputs
    auto output_shape = at::infer_size(condition.sizes(), x.sizes());
    output_shape = at::infer_size(output_shape, y.sizes());

    // Expand inputs to the broadcasted shape and ensure contiguity
    // This simplifies the kernel, allowing it to operate on flattened memory
    condition = condition.expand(output_shape).contiguous();
    x = x.expand(output_shape).contiguous();
    y = y.expand(output_shape).contiguous();
    
    int64_t num_elements = x.numel();

    // Create the output tensor with the broadcasted shape and x's options (type, device)
    auto out = torch::empty(output_shape, x.options());

    // Determine grid and block dimensions
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    // Manual type dispatch
    if (x.scalar_type() == torch::kFloat32) {
        where_kernel<float><<<num_blocks, threads_per_block>>>(
            condition.data_ptr<bool>(),
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            out.data_ptr<float>(),
            num_elements
        );
    } else if (x.scalar_type() == torch::kFloat64) {
        where_kernel<double><<<num_blocks, threads_per_block>>>(
            condition.data_ptr<bool>(),
            x.data_ptr<double>(),
            y.data_ptr<double>(),
            out.data_ptr<double>(),
            num_elements
        );
    } else if (x.scalar_type() == torch::kHalf) {
        where_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            condition.data_ptr<bool>(),
            x.data_ptr<c10::Half>(),
            y.data_ptr<c10::Half>(),
            out.data_ptr<c10::Half>(),
            num_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for X and Y tensors in where_op");
    }

    // Check for CUDA kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return out;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_where, "Element-wise where operation on CUDA tensors.");
}
// [END kernel.cu]