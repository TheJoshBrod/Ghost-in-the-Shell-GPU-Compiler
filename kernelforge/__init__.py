"""KernelForge runtime loading helpers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernelforge.run_cast import CastModelRuntime


_RUNTIME_EXPORTS = {"CastModelRuntime", "get_runtime_stats", "load_cast", "reset_runtime_stats"}


def __getattr__(name: str):
    if name in _RUNTIME_EXPORTS:
        module = import_module("kernelforge.run_cast")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load(
    cast_path: str,
    *,
    device: str | None = None,
    model_args: dict | None = None,
    no_kernels: bool = False,
    opt_level: str = "-O3",
) -> CastModelRuntime:
    from kernelforge.run_cast import load_cast

    return load_cast(
        cast_path,
        model_args=model_args,
        no_kernels=no_kernels,
        opt_level=opt_level,
        device=device,
    )


__all__ = ["CastModelRuntime", "get_runtime_stats", "load", "load_cast", "reset_runtime_stats"]
