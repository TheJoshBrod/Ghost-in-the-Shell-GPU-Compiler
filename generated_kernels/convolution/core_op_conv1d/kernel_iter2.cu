// [START kernel.cu]
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for 1D convolution
// Input: N x C x L, Filter: K x C x S, Bias: K
template <typename T>
__global__ void conv1d_kernel(const T* __restrict__ input, const T* __restrict__ filter, const T* __restrict__ bias, T* __restrict__ output, int64_t N, int64_t C, int64_t L, int64_t K, int64_t S, int64_t L_out) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < N * K * L_out; i += stride) {
        int64_t n = i / (K * L_out);
        int64_t k = (i / L_out) % K;
        int64_t l_out = i % L_out;

        float result = static_cast<float>(bias[k]);
        for (int64_t c = 0; c < C; c++) {
            for (int64_t s = 0; s < S; s++) {
                int64_t l_in = l_out + s;
                if (l_in < L) {
                    result += static_cast<float>(input[(n * C + c) * L + l_in]) * static_cast<float>(filter[(k * C + c) * S + s]);
                }
            }
        }
        output[i] = static_cast<T>(result);
    }
}

// Wrapper function
torch::Tensor launch_conv1d(torch::Tensor input, torch::Tensor filter, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(filter.is_cuda(), "Filter must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3 && filter.dim() == 3 && bias.dim() == 1, "Incorrect dimensions");
    TORCH_CHECK(input.size(1) == filter.size(1), "Channel mismatch");
    TORCH_CHECK(filter.size(0) == bias.size(0), "Kernel mismatch");

    input = input.contiguous();
    filter = filter.contiguous();
    bias = bias.contiguous();

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);
    int64_t K = filter.size(0);
    int64_t S = filter.size(2);

    int64_t L_out = L - S + 1;
    auto output = torch::empty({N, K, L_out}, input.options());

    int threads = 256;
    int blocks = (N * K * L_out + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "conv1d_kernel", ([&] {
        conv1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), N, C, L, K, S, L_out
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    
    return output;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_conv1d, "1D Convolution kernel");
}
// [END kernel.cu]