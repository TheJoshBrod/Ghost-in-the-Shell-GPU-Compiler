// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for 2D convolution operation
template <typename T>
__global__ void conv2d_kernel(const T* input, const T* kernel, T* output, int64_t N, int64_t C, int64_t H, int64_t W, int64_t K, int64_t KH, int64_t KW, int64_t OH, int64_t OW) {
    int64_t n = blockIdx.x;
    int64_t c_o = blockIdx.y;
    int64_t h = blockIdx.z * blockDim.x + threadIdx.x;
    int64_t w = blockIdx.z * blockDim.y + threadIdx.y;

    if (h < OH && w < OW) {
        T sum = 0;
        for (int64_t c_i = 0; c_i < C; ++c_i) {
            for (int64_t kh = 0; kh < KH; ++kh) {
                for (int64_t kw = 0; kw < KW; ++kw) {
                    int64_t h_in = h + kh;
                    int64_t w_in = w + kw;
                    if (h_in < H && w_in < W) {
                        sum += input[((n * C + c_i) * H + h_in) * W + w_in] * kernel[((c_o * C + c_i) * KH + kh) * KW + kw];
                    }
                }
            }
        }
        output[(n * K + c_o) * OH * OW + h * OW + w] = sum;
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function
torch::Tensor launch_conv2d(torch::Tensor input, torch::Tensor kernel) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(kernel.is_cuda(), "kernel must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be 4D tensor");
    TORCH_CHECK(kernel.dim() == 4, "kernel must be 4D tensor");

    input = input.contiguous();
    kernel = kernel.contiguous();

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t K = kernel.size(0);
    int64_t KH = kernel.size(2);
    int64_t KW = kernel.size(3);
    int64_t OH = H - KH + 1;
    int64_t OW = W - KW + 1;

    auto output = torch::empty({N, K, OH, OW}, input.options());

    dim3 grid(N, K, (OH * OW + 255) / 256);
    dim3 block(256);

    if (input.scalar_type() == torch::kFloat32) {
        conv2d_kernel<float><<<grid, block>>>(
            input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, K, KH, KW, OH, OW
        );
    } else if (input.scalar_type() == torch::kFloat64) {
        conv2d_kernel<double><<<grid, block>>>(
            input.data_ptr<double>(), kernel.data_ptr<double>(), output.data_ptr<double>(), N, C, H, W, K, KH, KW, OH, OW
        );
    } else if (input.scalar_type() == torch::kHalf) {
        conv2d_kernel<c10::Half><<<grid, block>>>(
            input.data_ptr<c10::Half>(), kernel.data_ptr<c10::Half>(), output.data_ptr<c10::Half>(), N, C, H, W, K, KH, KW, OH, OW
        );
    } else {
        TORCH_CHECK(false, "Unsupported type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return output;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_conv2d, "2D Convolution kernel");
}
// [END kernel.cu]