// fused_gemm_sin.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(err) do { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

constexpr int TILE = 16;

extern "C"
__global__ void fused_sgemm_sin(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = by * TILE + ty;
    const int col = bx * TILE + tx;

    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        int a_col = k0 + tx;
        int b_row = k0 + ty;

        if (row < M && a_col < K) {
            sA[ty][tx] = A[row * K + a_col];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if (b_row < K && col < N) {
            sB[ty][tx] = B[b_row * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            acc += sA[ty][t] * sB[t][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sinf(acc);
    }
}

int main() {
    const int M = 10, K = 10, N = 10;

    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    float *hA = (float*)malloc(sizeA * sizeof(float));
    float *hB = (float*)malloc(sizeB * sizeof(float));
    float *hC = (float*)malloc(sizeC * sizeof(float));

    for (int i = 0; i < sizeA; i++) hA[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < sizeB; i++) hB[i] = (float)rand() / RAND_MAX;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, sizeC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeB * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    fused_sgemm_sin<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Fused GEMM+sin (first 6 vals):\n");
    for (int i = 0; i < 6 && i < sizeC; i++) {
        printf("% .6f ", hC[i]);
    }
    printf("\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
