// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>

// ============ DEVICE CODE (BEFORE PyTorch headers) ============

// CUDA kernel for 1D convolution
// input: batch_size x in_channels x in_width
// weight: out_channels x in_channels x kernel_width
// bias: out_channels
// output: batch_size x out_channels x out_width
template <typename T>
__global__ void conv1d_kernel(
    const T* input, const T* weight, const T* bias, T* output,
    int64_t batch_size, int64_t in_channels, int64_t in_width,
    int64_t out_channels, int64_t kernel_width, int64_t out_width) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = batch_size * out_channels * out_width;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t w_out = i % out_width;
        int64_t c_out = (i / out_width) % out_channels;
        int64_t b = i / (out_channels * out_width);

        float value = static_cast<float>(bias[c_out]);

        for (int64_t c_in = 0; c_in < in_channels; ++c_in) {
            for (int64_t k = 0; k < kernel_width; ++k) {
                int64_t w_in = w_out + k;
                if (w_in < in_width) {
                    value += static_cast<float>(input[b * in_channels * in_width + c_in * in_width + w_in]) *
                             static_cast<float>(weight[c_out * in_channels * kernel_width + c_in * kernel_width + k]);
                }
            }
        }
        output[i] = static_cast<T>(value);
    }
}

// ============ HOST CODE (AFTER PyTorch headers) ============
#include <torch/extension.h>

// Wrapper function for 1D convolution
torch::Tensor launch_conv1d(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {

    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (batch_size, in_channels, in_width)");
    TORCH_CHECK(weight.dim() == 3, "Weight must be 3D (out_channels, in_channels, kernel_width)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D (out_channels)");

    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_width = input.size(2);
    int64_t out_channels = weight.size(0);
    int64_t kernel_width = weight.size(2);
    int64_t out_width = in_width - kernel_width + 1;

    auto output = torch::empty({batch_size, out_channels, out_width}, input.options());

    int threads = 256;
    int blocks = (batch_size * out_channels * out_width + threads - 1) / threads;

    if (input.scalar_type() == torch::kFloat32) {
        conv1d_kernel<float><<<blocks, threads>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
            batch_size, in_channels, in_width, out_channels, kernel_width, out_width
        );
    } else if (input.scalar_type() == torch::kFloat64) {
        conv1d_kernel<double><<<blocks, threads>>>(
            input.data_ptr<double>(), weight.data_ptr<double>(), bias.data_ptr<double>(), output.data_ptr<double>(),
            batch_size, in_channels, in_width, out_channels, kernel_width, out_width
        );
    } else if (input.scalar_type() == torch::kHalf) {
        conv1d_kernel<c10::Half><<<blocks, threads>>>(
            input.data_ptr<c10::Half>(), weight.data_ptr<c10::Half>(), bias.data_ptr<c10::Half>(), output.data_ptr<c10::Half>(),
            batch_size, in_channels, in_width, out_channels, kernel_width, out_width
        );
    } else {
        TORCH_CHECK(false, "Unsupported datatype");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return output;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_conv1d, "1D Convolution");
}
// [END kernel.cu]