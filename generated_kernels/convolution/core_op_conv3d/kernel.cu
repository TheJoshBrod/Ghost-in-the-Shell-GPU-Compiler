// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/util/Half.h>

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for 3D convolution
// Assumes:
// - A: input tensor with shape [B, C_in, D_in, H_in, W_in]
// - W: weights tensor with shape [C_out, C_in, KD, KH, KW]
// - B: bias tensor with shape [C_out]
// - Output: output tensor with shape [B, C_out, D_out, H_out, W_out]
template <typename T>
__global__ void conv3d_kernel(const T* A, const T* W, const T* B, T* Output,
                              int64_t B_size, int64_t C_in, int64_t D_in, int64_t H_in, int64_t W_in,
                              int64_t C_out, int64_t KD, int64_t KH, int64_t KW,
                              int64_t D_out, int64_t H_out, int64_t W_out) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = B_size * C_out * D_out * H_out * W_out;

    for (int64_t linear_idx = idx; linear_idx < total; linear_idx += stride) {
        int64_t w = linear_idx % W_out;
        int64_t h = (linear_idx / W_out) % H_out;
        int64_t d = (linear_idx / (W_out * H_out)) % D_out;
        int64_t cout = (linear_idx / (W_out * H_out * D_out)) % C_out;
        int64_t b = linear_idx / (W_out * H_out * D_out * C_out);

        T value = B[cout];
        for (int64_t cin = 0; cin < C_in; ++cin) {
            for (int64_t kd = 0; kd < KD; ++kd) {
                for (int64_t kh = 0; kh < KH; ++kh) {
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        int64_t id = d + kd;
                        int64_t ih = h + kh;
                        int64_t iw = w + kw;
                        if (id < D_in && ih < H_in && iw < W_in) {
                            value += A[((b * C_in + cin) * D_in + id) * H_in * W_in + ih * W_in + iw] *
                                     W[((cout * C_in + cin) * KD + kd) * KH * KW + kh * KW + kw];
                        }
                    }
                }
            }
        }
        Output[linear_idx] = value;
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function
torch::Tensor launch_conv3d(torch::Tensor A, torch::Tensor W, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 5 && W.dim() == 5, "Inputs must be 5D");
    TORCH_CHECK(B.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(A.size(1) == W.size(1), "Input channels must match");

    // Ensure tensors are contiguous
    A = A.contiguous();
    W = W.contiguous();
    B = B.contiguous();

    int64_t B_size = A.size(0);
    int64_t C_in = A.size(1);
    int64_t D_in = A.size(2);
    int64_t H_in = A.size(3);
    int64_t W_in = A.size(4);
    
    int64_t C_out = W.size(0);
    int64_t KD = W.size(2);
    int64_t KH = W.size(3);
    int64_t KW = W.size(4);
    
    int64_t D_out = D_in - KD + 1;
    int64_t H_out = H_in - KH + 1;
    int64_t W_out = W_in - KW + 1;

    auto output = torch::empty({B_size, C_out, D_out, H_out, W_out}, A.options());

    int threads = 256;
    int blocks = (B_size * C_out * D_out * H_out * W_out + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "conv3d_kernel_launch", ([&] {
        conv3d_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(), W.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            B_size, C_in, D_in, H_in, W_in, C_out, KD, KH, KW, D_out, H_out, W_out
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return output;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_conv3d, "3D Convolution kernel");
}
// [END kernel.cu]