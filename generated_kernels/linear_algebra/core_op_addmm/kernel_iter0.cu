// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for addmm operation: output = beta * input + alpha * (mat1 @ mat2)
// input: M x N
// mat1: M x K
// mat2: K x N
// output: M x N
template <typename T>
__global__ void addmm_kernel(
    const T* input_ptr,
    const T* mat1_ptr,
    const T* mat2_ptr,
    T* output_ptr,
    int64_t M,
    int64_t K,
    int64_t N,
    T beta,
    T alpha) {

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t total_elements = M * N;

    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t row = i / N;
        int64_t col = i % N;

        // Compute the matrix multiplication part (mat1 @ mat2)
        float sum_mm = 0.0f; // Use float for accumulation to maintain precision
        for (int64_t k = 0; k < K; k++) {
            sum_mm += static_cast<float>(mat1_ptr[row * K + k]) * static_cast<float>(mat2_ptr[k * N + col]);
        }

        // Compute the final result: beta * input + alpha * sum_mm
        float input_val = static_cast<float>(input_ptr[i]);
        float output_val_f = static_cast<float>(beta) * input_val + static_cast<float>(alpha) * sum_mm;
        
        output_ptr[i] = static_cast<T>(output_val_f);
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// C++ wrapper function to launch the CUDA kernel
torch::Tensor launch_addmm(
    torch::Tensor input,
    torch::Tensor mat1,
    torch::Tensor mat2,
    double beta_d,  // scalar beta from Python, typically double
    double alpha_d  // scalar alpha from Python, typically double
) {
    // --- Tensor Checks ---
    TORCH_CHECK(input.is_cuda(), "Input tensor must be CUDA tensor.");
    TORCH_CHECK(mat1.is_cuda(), "Mat1 tensor must be CUDA tensor.");
    TORCH_CHECK(mat2.is_cuda(), "Mat2 tensor must be CUDA tensor.");

    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(mat1.dim() == 2, "Mat1 tensor must be 2D.");
    TORCH_CHECK(mat2.dim() == 2, "Mat2 tensor must be 2D.");

    TORCH_CHECK(input.scalar_type() == mat1.scalar_type(), "Input and Mat1 tensors must have the same data type.");
    TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "Mat1 and Mat2 tensors must have the same data type.");

    // Dimension checks for matrix multiplication (mat1 @ mat2)
    TORCH_CHECK(mat1.size(1) == mat2.size(0),
                "Mat1's second dimension must match Mat2's first dimension for matrix multiplication.");

    // Dimension checks for input addition (output = beta * input + ...)
    TORCH_CHECK(input.size(0) == mat1.size(0),
                "Input's first dimension must match Mat1's first dimension for element-wise addition.");
    TORCH_CHECK(input.size(1) == mat2.size(1),
                "Input's second dimension must match Mat2's second dimension for element-wise addition.");

    // Ensure tensors are contiguous for efficient memory access
    input = input.contiguous();
    mat1 = mat1.contiguous();
    mat2 = mat2.contiguous();

    // Determine dimensions
    int64_t M = mat1.size(0);
    int64_t K = mat1.size(1);
    int64_t N = mat2.size(1);

    // Create output tensor with the same options (device, dtype) as input
    auto output = torch::empty({M, N}, input.options());

    // --- Kernel Launch Configuration ---
    int threads_per_block = 256;
    int blocks_per_grid = (M * N + threads_per_block - 1) / threads_per_block;

    // --- Type Dispatch ---
    // Launch kernel based on the tensor's scalar type
    if (input.scalar_type() == torch::kFloat32) {
        float beta_f = static_cast<float>(beta_d);
        float alpha_f = static_cast<float>(alpha_d);
        addmm_kernel<float><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<float>(),
            mat1.data_ptr<float>(),
            mat2.data_ptr<float>(),
            output.data_ptr<float>(),
            M, K, N,
            beta_f, alpha_f
        );
    } else if (input.scalar_type() == torch::kFloat64) {
        // No conversion needed, double to double
        double beta_f = beta_d;
        double alpha_f = alpha_d;
        addmm_kernel<double><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<double>(),
            mat1.data_ptr<double>(),
            mat2.data_ptr<double>(),
            output.data_ptr<double>(),
            M, K, N,
            beta_f, alpha_f
        );
    } else if (input.scalar_type() == torch::kHalf) {
        // Convert double scalars to c10::Half for the kernel
        c10::Half beta_h = static_cast<c10::Half>(beta_d);
        c10::Half alpha_h = static_cast<c10::Half>(alpha_d);
        addmm_kernel<c10::Half><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<c10::Half>(),
            mat1.data_ptr<c10::Half>(),
            mat2.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            M, K, N,
            beta_h, alpha_h
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type for addmm operation. Only float32, float64, and float16 (half) are supported.");
    }

    // --- CUDA Error Check ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}

// ============ PYBIND11 BINDING ============
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_addmm, "Custom CUDA kernel for aten::addmm operation.");
}
// [END kernel.cu]