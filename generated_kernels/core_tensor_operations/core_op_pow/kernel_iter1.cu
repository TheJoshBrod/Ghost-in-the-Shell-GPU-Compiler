// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cuda_fp16.h> // Required for __half type and half-precision intrinsics
#include <c10/util/Half.h> // Required for c10::Half when using data_ptr<c10::Half>()

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise power operation: out[i] = base[i] ** exp[i]
template <typename T>
__global__ void pow_kernel(const T* base, const T* exp, T* out, int64_t n_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        if constexpr (std::is_same_v<T, float>) {
            out[i] = powf(base[i], exp[i]);
        } else if constexpr (std::is_same_v<T, double>) {
            out[i] = pow(base[i], exp[i]);
        } else if constexpr (std::is_same_v<T, __half>) {
            // Convert to float for computation to maintain precision, then convert back
            float base_f = __half2float(base[i]);
            float exp_f = __half2float(exp[i]);
            out[i] = __float2half(powf(base_f, exp_f));
        }
        // Unsupported types are checked in the host wrapper, so no need for device-side error.
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Host wrapper function for the element-wise power operation
torch::Tensor launch_pow(torch::Tensor base, torch::Tensor exp) {
    // Critical checks for tensor arguments
    TORCH_CHECK(base.is_cuda(), "Input 'base' tensor must be on a CUDA device.");
    TORCH_CHECK(exp.is_cuda(), "Input 'exp' tensor must be on a CUDA device.");
    TORCH_CHECK(base.scalar_type() == exp.scalar_type(), "Input tensors 'base' and 'exp' must have the same data type.");
    TORCH_CHECK(base.sizes() == exp.sizes(), "Input tensors 'base' and 'exp' must have identical shapes for element-wise power (broadcasting not implemented in this kernel).");

    // Ensure tensors are contiguous for efficient memory access in the kernel
    base = base.contiguous();
    exp = exp.contiguous();

    // Determine the total number of elements
    int64_t n_elements = base.numel();

    // Create the output tensor with the same shape and options as the base tensor
    auto out = torch::empty_like(base);

    // CUDA launch configuration
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // Manual type dispatch to call the appropriate kernel instantiation
    if (base.scalar_type() == torch::kFloat32) {
        pow_kernel<float><<<num_blocks, threads_per_block>>>(
            base.data_ptr<float>(), exp.data_ptr<float>(), out.data_ptr<float>(), n_elements
        );
    } else if (base.scalar_type() == torch::kFloat64) {
        pow_kernel<double><<<num_blocks, threads_per_block>>>(
            base.data_ptr<double>(), exp.data_ptr<double>(), out.data_ptr<double>(), n_elements
        );
    } else if (base.scalar_type() == torch::kHalf) {
        // c10::Half on host maps to __half on device
        pow_kernel<__half><<<num_blocks, threads_per_block>>>(
            base.data_ptr<c10::Half>(), exp.data_ptr<c10::Half>(), out.data_ptr<c10::Half>(), n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type for element-wise power operation. Only float32, float64, and half are supported.");
    }

    // Critical: Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return out;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Expose the launch_pow function to Python as 'launch'
    m.def("launch", &launch_pow, "Element-wise power operation (base ** exp) on CUDA tensors.");
}
// [END kernel.cu]