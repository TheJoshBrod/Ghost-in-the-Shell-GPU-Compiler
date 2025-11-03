// [START kernel.cu]
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 2D convolution operation
template <typename T>
__global__ void conv2d_kernel(const T* input, const T* kernel, T* output, int64_t N, int64_t C, int64_t H, int64_t W, int64_t K, int64_t KH, int64_t KW, int64_t OH, int64_t OW) {
    int64_t n = blockIdx.x;
    int64_t c_o = blockIdx.y;
    int64_t pixelIdx = blockIdx.z * blockDim.x + threadIdx.x;
    int64_t h = pixelIdx / OW;
    int64_t w = pixelIdx % OW;

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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "conv2d_kernel", ([&] {
        conv2d_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(), kernel.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), N, C, H, W, K, KH, KW, OH, OW
        );
    }));

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaDeviceSynchronize());
    return output;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_conv2d, "2D Convolution kernel");
}
// [END kernel.cu]