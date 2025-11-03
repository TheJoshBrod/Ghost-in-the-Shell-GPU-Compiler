// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for element-wise subtraction: output = input1 - alpha * input2
template <typename T>
__global__ void sub_kernel(const T* input1, const T* input2, T* output, int64_t n_elements, double alpha_scalar_double) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        if (std::is_same<T, __half>::value) {
            // For half-precision, perform computation in float
            float val1 = static_cast<float>(input1[i]);
            float val2 = static_cast<float>(input2[i]);
            float alpha_val = static_cast<float>(alpha_scalar_double); // Cast alpha to float
            output[i] = static_cast<__half>(val1 - alpha_val * val2);
        } else if (std::is_same<T, float>::value) {
            // For float-precision, perform computation in float
            float val1 = static_cast<float>(input1[i]);
            float val2 = static_cast<float>(input2[i]);
            float alpha_val = static_cast<float>(alpha_scalar_double); // Cast alpha to float
            output[i] = static_cast<float>(val1 - alpha_val * val2);
        } else if (std::is_same<T, double>::value) {
            // For double-precision, perform computation in double
            double val1 = static_cast<double>(input1[i]);
            double val2 = static_cast<double>(input2[i]);
            double alpha_val = alpha_scalar_double; // Alpha is already double
            output[i] = static_cast<double>(val1 - alpha_val * val2);
        }
        // Other types (e.g., int, long) are not typically handled by aten::sub with float alpha.
        // If they were, their specific casting and computation logic would be added here.
    }
}


// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

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
    input1 = input1.contiguous();
    input2 = input2.contiguous();

    // --- Output Tensor Creation ---
    int64_t n_elements = input1.numel();
    auto output = torch::empty_like(input1); // Output tensor has same shape and data type as input1

    // --- Kernel Launch Configuration ---
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    // Extract scalar alpha value, converted to double for maximum precision
    double alpha_val_double = alpha.to<double>();

    // --- Type Dispatch and Kernel Launch ---
    if (input1.scalar_type() == torch::kFloat32) {
        sub_kernel<float><<<num_blocks, threads_per_block>>>(
            input1.data_ptr<float>(),
            input2.data_ptr<float>(),
            output.data_ptr<float>(),
            n_elements,
            alpha_val_double
        );
    } else if (input1.scalar_type() == torch::kFloat64) {
        sub_kernel<double><<<num_blocks, threads_per_block>>>(
            input1.data_ptr<double>(),
            input2.data_ptr<double>(),
            output.data_ptr<double>(),
            n_elements,
            alpha_val_double
        );
    } else if (input1.scalar_type() == torch::kHalf) {
        sub_kernel<c10::Half><<<num_blocks, threads_per_block>>>(
            input1.data_ptr<c10::Half>(),
            input2.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            n_elements,
            alpha_val_double
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type for element-wise subtraction! Supported types are float32, float64, and half.");
    }

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