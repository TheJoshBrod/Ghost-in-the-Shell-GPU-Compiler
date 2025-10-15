# Prompts for Generating Kernels

## Directly from Aten to CUDA Prompt

Your task is to translate high-level operator descriptions into functional CUDA kernel code.

You will receive text containing:
- "Aggregated Operator Performance" (e.g., aten::mm, aten::sin) 
- "Detailed CUDA Kernel Breakdown" (e.g. volta_sgemm_128x64_nn, void at::native::vectorized_elementwise_kernel<4, at::native::sin_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::sin_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>))

Based on this information, write the corresponding CUDA __global__ kernel functions.

Critically Important Rules:

Your response MUST contain ONLY the raw CUDA C++ code.

Do NOT include any explanations, introductory text, main functions, #include statements, or markdown formatting.

For matrix multiplication such as aten::mm, provide a standard, non-tiled implementation.

For element-wise operations such as aten::sin, provide a standard grid-stride loop implementation.
