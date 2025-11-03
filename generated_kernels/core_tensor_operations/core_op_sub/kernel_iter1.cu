// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Half.h> // Required for c10::Half
#include <type_traits> // Required for std::is_same

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise subtraction: output = input1 - alpha * input2
template <typename T>
__global__ void sub_kernel(const T* input1, const T* input2, T* output, int64_t n_elements, double alpha_scalar_double) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        // Use std::is_same with c10::Half for PyTorch half-precision type
        if (std::is_same<T, c10::Half>::value) {
            // For half-precision, perform computation in float
            float val1 = static_cast<float>(input1[i]);
            float val2 = static_cast<float>(input2[i]);
            float alpha_val = static_cast<float>(alpha_scalar_double); // Cast alpha to float
            output[i] = static_cast<c10::Half>(val1 - alpha_val * val2);
        } else if (std::is_same<T, float>::value) {
            // For float-precision, perform computation in float
            output[i] = input1[i] - static_cast<float>(alpha_scalar_double) * input2[i];
        } else if (std::is_same<T, double>::value) {
            // For double-precision, perform computation in double
            output[i] = input1[i] - static_cast<double>(alpha_scalar_double) * input2[i];
        }
    }
}


// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>
#include <ATen/Dispatch.h> // Required for AT_DISPATCH_FLOATING_TYPES_AND_HALF

// Wrapper function for element-wise subtraction
torch::Tensor launch_sub(torch::Tensor input1, torch::Tensor input2, torch::Scalar alpha) {
    // --- Input Checks ---
    TORCH_CHECK(input1.is_cuda(), "Input tensor 1 must be a CUDA tensor!");
    TORCH_CHECK(input2.is_cuda(), "Input tensor 2 must be a CUDA tensor!");
    TORCH_CHECK(input1.scalar_type() == input2.scalar_type(), "Input tensors must have the same data type!");
    
    // For this example, assuming strict element-wise operation, so shapes must match.
    // Broadcasting would require more complex shape logic.
    TORCH_CHECK(input1.sizes().equals(input2.sizes()), "Input tensors must have the same shape for element-wise subtraction!");

    // Ensure tensors are contiguous for efficient data access in the kernel
    // It's good practice to make copies if not contiguous or handle non-contiguity
    // in the kernel. For simplicity here, we ensure contiguity.
    if (!input1.is_contiguous()) {
        input1 = input1.contiguous();
    }
    if (!input2.is_contiguous()) {
        input2 = input2.contiguous();
    }

    // --- Output Tensor Creation ---
    int64_t n_elements = input1.numel();
    auto output = torch::empty_like(input1); // Output tensor has same shape and data type as input1

    // --- Kernel Launch Configuration ---
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;
    // Handle empty tensors to avoid launching kernel with 0 blocks if n_elements is 0
    if (n_elements == 0) { 
        num_blocks = 0;
    }

    // Extract scalar alpha value, converted to double for maximum precision
    double alpha_val_double = alpha.to<double>();

    // --- Type Dispatch and Kernel Launch ---
    // Use AT_DISPATCH_FLOATING_TYPES_AND_HALF for cleaner type dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "sub_kernel", [&] {
        if (num_blocks > 0) { // Only launch kernel if there are elements to process
            sub_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
                input1.data_ptr<scalar_t>(),
                input2.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n_elements,
                alpha_val_double
            );
        }
    });

    // --- CUDA Error Checking ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return output;
}

// --- Pybind11 Module Definition ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_sub, "Element-wise subtraction: output = input1 - alpha * input2");
}
// [END kernel.cu]