# ResNet-50 Residual Add and MaxPool Fix Notes

## Runtime Adapter Fixes

- `torch.nn.functional.max_pool2d` cast dispatch now reconstructs the full
  launch ABI: `input`, `kernel_size`, `stride`, `padding`, `dilation`,
  `ceil_mode`, and `return_indices`.
- Residual adds are profiled as `torch.tensor.iadd`, normalized to the operator
  directory `torch_tensor_iadd`, and deployed by a scoped `Tensor.__iadd__`
  runtime patch while the cast model executes.

## Validation

- MaxPool smoke run:
  `paper_benchmarks/runs/resnet50_max_pool_adapter_fix_smoke`
- MaxPool smoke status: correctness passed, 0 fallbacks, successful Forge
  launches for bs1 and bs32.
- Residual-add profiling status: captured 16 `torch_tensor_iadd` entries.
- Residual-add zero-shot generation: succeeded on attempt 2.
- Residual-add opt20: completed 20 optimization iterations.

## Residual-Add Per-Kernel Result

- PyTorch baseline: `0.004580799937 ms`
- Best opt20 Forge kernel: `0.003463359922 ms`
- Speedup: `1.322646228x`
- Best kernel: `kernels/projects/resnet50-imagenet1k-fp16-gb10/trees/torch_tensor_iadd/kernels/kernel_16.cu`
- Kernel source SHA256:
  `828cf2ecefc9d1ecd025f8c94462187ef03c5c24c0f3b22e148f733dcc89a480`

These are per-kernel/profile-replay results, not full end-to-end model results.
