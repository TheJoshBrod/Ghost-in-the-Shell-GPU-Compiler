from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import importlib
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .paths import project_dir_for_name
from .state import write_json_file

SKIP_FUNCTIONS = {
    "has_torch_function",
    "handle_torch_function",
    "is_storage",
    "result_type",
    "get_default_dtype",
}

# Only wrap functions that originate from these modules — anything else
# (dispatch guards from torch._C, JIT helpers, typing annotations, etc.) is skipped.
_COMPUTE_MODULES = frozenset({
    "torch",
    "torch.nn.functional",
    "torch._C._nn",
})

DEFAULT_SKIP_OPS = {
    "dropout",
    "dropout_",
    "alpha_dropout",
    "feature_alpha_dropout",
}

PROFILE_SKIP_OPS: set[str] = set(DEFAULT_SKIP_OPS)

calls: dict[str, list[dict[str, Any]]] = {}
_wrapped: set[Any] = set()
ENABLE_WRAPPING = True
CAPTURE_ACTIVE = False
skipped_counts: dict[str, int] = {}
unique_case_counts: dict[str, dict[str, int]] = {}
unique_case_metadata: dict[str, dict[str, dict[str, Any]]] = {}
TENSOR_IADD_FUNCTION_NAME = "torch.tensor.iadd"
_ORIGINAL_TENSOR_IADD = None
TENSOR_METHODS_TO_WRAP = {
    "as_strided",
    "contiguous",
    "expand",
    "expand_as",
    "flatten",
    "narrow",
    "permute",
    "reshape",
    "split",
    "squeeze",
    "t",
    "to",
    "transpose",
    "unsqueeze",
    "view",
}
_ORIGINAL_TENSOR_METHODS: dict[str, Any] = {}
_TENSOR_IADD_SIGNATURE = {
    "params": ["input", "other"],
    "defaults": {},
    "kinds": {
        "input": inspect.Parameter.POSITIONAL_OR_KEYWORD.name,
        "other": inspect.Parameter.POSITIONAL_OR_KEYWORD.name,
    },
}


def _serialize(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    if isinstance(v, (list, tuple)):
        return type(v)(_serialize(x) for x in v)
    if isinstance(v, dict):
        return {k: _serialize(x) for k, x in v.items()}
    return v  # torch.dtype, torch.device, int, float, bool, None, etc.


_UNIQUE_CAPTURE_VALUES = {"unique", "unique_cases", "unique-case", "unique_kernel_cases"}


def _profile_capture_mode() -> str:
    raw = os.environ.get("KFORGE_PROFILE_CAPTURE_MODE", "").strip().lower()
    if raw in _UNIQUE_CAPTURE_VALUES:
        return "unique"
    return "capped"


def _is_unique_capture_mode() -> bool:
    return _profile_capture_mode() == "unique"


def _memory_format_flags(tensor: torch.Tensor) -> list[str]:
    flags: list[str] = []
    try:
        if tensor.is_contiguous():
            flags.append("contiguous")
    except Exception:
        pass
    try:
        if tensor.dim() == 4 and tensor.is_contiguous(memory_format=torch.channels_last):
            flags.append("channels_last")
    except Exception:
        pass
    try:
        if tensor.dim() == 5 and tensor.is_contiguous(memory_format=torch.channels_last_3d):
            flags.append("channels_last_3d")
    except Exception:
        pass
    return flags


def _profile_signature_value(value: Any) -> Any:
    if torch.is_tensor(value):
        try:
            device_type = str(value.device.type)
        except Exception:
            device_type = ""
        try:
            storage_offset = int(value.storage_offset())
        except Exception:
            storage_offset = 0
        return {
            "type": "tensor",
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "device_type": device_type,
            "layout": str(value.layout),
            "stride": [int(dim) for dim in value.stride()],
            "storage_offset": storage_offset,
            "requires_grad": bool(value.requires_grad),
            "memory_format": _memory_format_flags(value),
        }
    if isinstance(value, torch.Size):
        return {"type": "torch.Size", "items": [int(dim) for dim in value]}
    if isinstance(value, torch.dtype):
        return {"type": "torch.dtype", "value": str(value)}
    if isinstance(value, torch.device):
        return {"type": "torch.device", "value": str(value)}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "items": [_profile_signature_value(item) for item in value],
        }
    if isinstance(value, dict):
        items = []
        for key in sorted(value.keys(), key=lambda item: str(item)):
            items.append([str(key), _profile_signature_value(value[key])])
        return {"type": "dict", "items": items}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__name__, "value": repr(value)}


def _profile_case_payload(function_name: str, args: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "function_name": function_name,
        "args": _profile_signature_value(args),
        "kwargs": _profile_signature_value(kwargs),
    }


def _profile_case_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_unique_case(function_name: str, payload: dict[str, Any]) -> tuple[str, bool]:
    case_key = _profile_case_key(payload)
    counts = unique_case_counts.setdefault(function_name, {})
    next_count = counts.get(case_key, 0) + 1
    counts[case_key] = next_count

    metadata = unique_case_metadata.setdefault(function_name, {})
    case_meta = metadata.setdefault(case_key, {"case_key": case_key, "signature": payload})
    case_meta["count"] = next_count
    return case_key, next_count == 1


def _annotate_unique_entry(entry: dict[str, Any], case_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    entry["profile_case_key"] = case_key
    entry["profile_case_count"] = 1
    entry["profile_case_signature"] = payload
    return entry


def _op_dir_name(function_name: str) -> str:
    return function_name.replace(".", "_").replace("/", "_")


def _unique_case_summary() -> dict[str, Any]:
    ops: dict[str, Any] = {}
    for function_name in sorted(unique_case_counts.keys()):
        case_counts = unique_case_counts[function_name]
        metadata = unique_case_metadata.get(function_name, {})
        cases: list[dict[str, Any]] = []
        for case_key in sorted(case_counts.keys()):
            case_meta = dict(metadata.get(case_key, {}))
            case_meta["case_key"] = case_key
            case_meta["count"] = int(case_counts[case_key])
            cases.append(case_meta)
        total_calls = sum(int(value) for value in case_counts.values())
        ops[function_name] = {
            "op_dir": _op_dir_name(function_name),
            "total_calls": int(total_calls),
            "unique_cases": len(case_counts),
            "duplicate_calls": int(total_calls - len(case_counts)),
            "cases": cases,
        }
    return {
        "capture_mode": "unique",
        "ops": ops,
    }


def _reset_profile_capture_state() -> None:
    calls.clear()
    skipped_counts.clear()
    unique_case_counts.clear()
    unique_case_metadata.clear()


# Known signatures for C-extension ops that lack Python-inspectable signatures.
# Matches the public PyTorch API parameter order exactly.
_KNOWN_SIGS: dict[str, dict] = {  # noqa: E501  (line-length; values kept readable)
    "torch.nn.functional.linear": {
        "params": ["input", "weight", "bias"],
        "defaults": {"bias": None},
    },
    "torch.nn.functional.conv1d": {
        "params": ["input", "weight", "bias", "stride", "padding", "dilation", "groups"],
        "defaults": {"bias": None, "stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    },
    "torch.nn.functional.conv2d": {
        "params": ["input", "weight", "bias", "stride", "padding", "dilation", "groups"],
        "defaults": {"bias": None, "stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    },
    "torch.nn.functional.conv3d": {
        "params": ["input", "weight", "bias", "stride", "padding", "dilation", "groups"],
        "defaults": {"bias": None, "stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    },
    "torch.nn.functional.conv_transpose1d": {
        "params": ["input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation"],
        "defaults": {"bias": None, "stride": 1, "padding": 0, "output_padding": 0, "groups": 1, "dilation": 1},
    },
    "torch.nn.functional.conv_transpose2d": {
        "params": ["input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation"],
        "defaults": {"bias": None, "stride": 1, "padding": 0, "output_padding": 0, "groups": 1, "dilation": 1},
    },
    "torch.nn.functional.max_pool2d": {
        "params": ["input", "kernel_size", "stride", "padding", "dilation", "ceil_mode", "return_indices"],
        "defaults": {"stride": None, "padding": 0, "dilation": 1, "ceil_mode": False, "return_indices": False},
    },
    "torch.nn.functional.avg_pool2d": {
        "params": ["input", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"],
        "defaults": {"stride": None, "padding": 0, "ceil_mode": False, "count_include_pad": True, "divisor_override": None},
    },
    "torch.nn.functional.adaptive_avg_pool2d": {
        "params": ["input", "output_size"],
        "defaults": {},
    },
    "torch.nn.functional.adaptive_max_pool2d": {
        "params": ["input", "output_size", "return_indices"],
        "defaults": {"return_indices": False},
    },
}


def _build_signature(op_dict: dict) -> inspect.Signature:
    parameters = []
    for name in op_dict["params"]:
        if name in op_dict["defaults"]:
            param = inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=op_dict["defaults"][name],
            )
        else:
            param = inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        parameters.append(param)
    return inspect.Signature(parameters)


# Pre-built Signature objects — computed once at import time.
_COMPILED_SIGS: dict[str, inspect.Signature] = {
    k: _build_signature(v) for k, v in _KNOWN_SIGS.items()
}


@contextmanager
def _patched_auto_docstring():
    restored = []
    try:
        auto_docstring_module = importlib.import_module("transformers.utils.auto_docstring")
        utils_module = importlib.import_module("transformers.utils")

        def _identity(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def _decorator(obj):
                return obj

            return _decorator

        original_mod = getattr(auto_docstring_module, "auto_docstring", None)
        if original_mod:
            restored.append((auto_docstring_module, "auto_docstring", original_mod))
            auto_docstring_module.auto_docstring = _identity

        original_utils = getattr(utils_module, "auto_docstring", None)
        if original_utils:
            restored.append((utils_module, "auto_docstring", original_utils))
            utils_module.auto_docstring = _identity
    except Exception:
        restored = []

    try:
        yield
    finally:
        for module, attr, original in restored:
            try:
                setattr(module, attr, original)
            except Exception:
                pass


def _normalize_op_name(full_key: str) -> str:
    return full_key.split(".")[-1].lower().strip()


def _load_profile_filters(_config: dict[str, Any]) -> None:
    global PROFILE_SKIP_OPS
    PROFILE_SKIP_OPS = set(DEFAULT_SKIP_OPS)


def _should_skip(full_key: str) -> bool:
    op_name = _normalize_op_name(full_key)
    if op_name in PROFILE_SKIP_OPS or full_key.lower() in PROFILE_SKIP_OPS:
        return True
    return False


def wrap_function(module, func_name: str) -> None:
    if not ENABLE_WRAPPING:
        return
    func = getattr(module, func_name)
    if func in _wrapped:
        return
    _wrapped.add(func)
    module_path = module.__name__

    # Precompute full signature once per wrapped function.
    # Priority: _COMPILED_SIGS (hardcoded, reliable) → inspect (Python-native ops).
    # Using module_path.func_name avoids the func.__module__ trap where C-extension
    # ops report their internal module (e.g. torch._C._nn) rather than the public one.
    op_key = f"{module_path}.{func_name}"
    _func_sig = _COMPILED_SIGS.get(op_key)
    _sig_params: list[str] = []
    _sig_defaults: dict[str, Any] = {}
    _sig_kinds: dict[str, str] = {}

    if _func_sig:
        _sig_params = list(_func_sig.parameters.keys())
        _sig_kinds = {
            k: v.kind.name
            for k, v in _func_sig.parameters.items()
        }
        _sig_defaults = {
            k: v.default
            for k, v in _func_sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
    else:
        try:
            _func_sig = inspect.signature(func)
            for name, param in _func_sig.parameters.items():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                _sig_params.append(name)
                _sig_kinds[name] = param.kind.name
                if param.default is not inspect.Parameter.empty:
                    _sig_defaults[name] = param.default
        except (ValueError, TypeError):
            pass
        if not _sig_params:
            # Stub (*args, **kwargs) or uninspectable — discard the signature object
            # so the fallback recording path is used in wrapper.
            _func_sig = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{module_path}.{func_name}"
        output = func(*args, **kwargs)

        if not CAPTURE_ACTIVE:
            return output
        if _should_skip(key):
            skipped_counts[key] = skipped_counts.get(key, 0) + 1
            return output

        unique_capture = _is_unique_capture_mode()

        # Try to resolve full parameter set including defaults.
        # _func_sig is always a real inspect.Signature when set, so bind() +
        # apply_defaults() handles both Python-native and C-extension ops uniformly.
        if _func_sig and _sig_params:
            try:
                bound = _func_sig.bind(*args, **kwargs)
                bound.apply_defaults()
                live_kwargs = dict(bound.arguments)
                case_key = ""
                case_payload: dict[str, Any] | None = None
                if unique_capture:
                    case_payload = _profile_case_payload(key, [], live_kwargs)
                    case_key, should_save = _record_unique_case(key, case_payload)
                    if not should_save:
                        duplicate_key = f"{key}:duplicate_profile_case"
                        skipped_counts[duplicate_key] = skipped_counts.get(duplicate_key, 0) + 1
                        return output
                else:
                    entries = calls.setdefault(key, [])
                    max_per_op = _profile_max_per_op()
                    if _profile_limit_reached(len(entries), max_per_op):
                        limit_key = f"{key}:profile_max_per_op"
                        skipped_counts[limit_key] = skipped_counts.get(limit_key, 0) + 1
                        return output

                entries = calls.setdefault(key, [])
                ser_output = _serialize(output)
                resolved_kwargs = {k: _serialize(v) for k, v in bound.arguments.items()}
                entry = {
                    "function_name": key,
                    "args": [],
                    "kwargs": resolved_kwargs,
                    "output": ser_output,
                    "signature": {"params": _sig_params, "defaults": _sig_defaults, "kinds": _sig_kinds},
                }
                if unique_capture and case_payload is not None:
                    _annotate_unique_entry(entry, case_key, case_payload)
                entries.append(entry)
                return output
            except TypeError:
                pass  # bind failed — fall through to original

        case_key = ""
        case_payload = None
        if unique_capture:
            case_payload = _profile_case_payload(key, list(args), dict(kwargs))
            case_key, should_save = _record_unique_case(key, case_payload)
            if not should_save:
                duplicate_key = f"{key}:duplicate_profile_case"
                skipped_counts[duplicate_key] = skipped_counts.get(duplicate_key, 0) + 1
                return output
        else:
            entries = calls.setdefault(key, [])
            max_per_op = _profile_max_per_op()
            if _profile_limit_reached(len(entries), max_per_op):
                limit_key = f"{key}:profile_max_per_op"
                skipped_counts[limit_key] = skipped_counts.get(limit_key, 0) + 1
                return output

        # Original recording path (fallback)
        entries = calls.setdefault(key, [])
        ser_output = _serialize(output)
        entry = {
            "function_name": key,
            "args": [_serialize(a) for a in args],
            "kwargs": {k: _serialize(v) for k, v in kwargs.items()},
            "output": ser_output,
        }
        if unique_capture and case_payload is not None:
            _annotate_unique_entry(entry, case_key, case_payload)
        entries.append(entry)
        return output

    setattr(module, func_name, wrapper)


def _wrap_module_callables(module) -> None:
    for name in dir(module):
        if name.startswith("_"):
            continue
        if name in SKIP_FUNCTIONS:
            continue
        obj = getattr(module, name)
        if inspect.isclass(obj):
            continue
        if not callable(obj):
            continue
        if (getattr(obj, "__module__", "") or "") not in _COMPUTE_MODULES:
            continue
        wrap_function(module, name)


def wrap_torch_ops() -> None:
    _wrap_module_callables(torch)
    _wrap_module_callables(F)
    try:
        _wrap_module_callables(torch._C._nn)
    except Exception:
        pass
    wrap_tensor_methods()


def _clone_for_profile(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    return value


def _tensor_method_function_name(method_name: str) -> str:
    return f"torch.tensor.{method_name}"


def wrap_tensor_method(method_name: str) -> None:
    if not ENABLE_WRAPPING or method_name in _ORIGINAL_TENSOR_METHODS:
        return
    original = getattr(torch.Tensor, method_name, None)
    if original is None or not callable(original):
        return

    key = _tensor_method_function_name(method_name)

    @wraps(original)
    def wrapper(self, *args, **kwargs):
        output = original(self, *args, **kwargs)

        if not CAPTURE_ACTIVE:
            return output
        if _should_skip(key):
            skipped_counts[key] = skipped_counts.get(key, 0) + 1
            return output

        unique_capture = _is_unique_capture_mode()
        entries = calls.setdefault(key, [])
        max_per_op = _profile_max_per_op()
        if not unique_capture and _profile_limit_reached(len(entries), max_per_op):
            limit_key = f"{key}:profile_max_per_op"
            skipped_counts[limit_key] = skipped_counts.get(limit_key, 0) + 1
            return output

        live_args = [self, *args]
        case_payload = None
        case_key = ""
        if unique_capture:
            case_payload = _profile_case_payload(key, live_args, dict(kwargs))
            case_key, should_save = _record_unique_case(key, case_payload)
            if not should_save:
                duplicate_key = f"{key}:duplicate_profile_case"
                skipped_counts[duplicate_key] = skipped_counts.get(duplicate_key, 0) + 1
                return output

        entry = {
            "function_name": key,
            "args": [_serialize(arg) for arg in live_args],
            "kwargs": {k: _serialize(v) for k, v in kwargs.items()},
            "output": _serialize(output),
        }
        if unique_capture and case_payload is not None:
            _annotate_unique_entry(entry, case_key, case_payload)
        entries.append(entry)
        return output

    _ORIGINAL_TENSOR_METHODS[method_name] = original
    setattr(torch.Tensor, method_name, wrapper)


def wrap_tensor_methods() -> None:
    for method_name in sorted(TENSOR_METHODS_TO_WRAP):
        try:
            wrap_tensor_method(method_name)
        except Exception:
            continue


def _restore_tensor_method_wrappers() -> None:
    for method_name, original in list(_ORIGINAL_TENSOR_METHODS.items()):
        try:
            setattr(torch.Tensor, method_name, original)
        except Exception:
            pass
    _ORIGINAL_TENSOR_METHODS.clear()


def wrap_tensor_iadd() -> None:
    global _ORIGINAL_TENSOR_IADD
    if not ENABLE_WRAPPING or _ORIGINAL_TENSOR_IADD is not None:
        return

    original = torch.Tensor.__iadd__
    _ORIGINAL_TENSOR_IADD = original

    def wrapper(self, other):
        lhs_before = None
        other_before = None
        unique_capture = _is_unique_capture_mode()
        if CAPTURE_ACTIVE and not _should_skip(TENSOR_IADD_FUNCTION_NAME):
            entries = calls.setdefault(TENSOR_IADD_FUNCTION_NAME, [])
            max_per_op = _profile_max_per_op()
            if unique_capture or not _profile_limit_reached(len(entries), max_per_op):
                lhs_before = _clone_for_profile(self)
                other_before = _clone_for_profile(other)

        output = original(self, other)

        if not CAPTURE_ACTIVE:
            return output
        if _should_skip(TENSOR_IADD_FUNCTION_NAME):
            skipped_counts[TENSOR_IADD_FUNCTION_NAME] = skipped_counts.get(TENSOR_IADD_FUNCTION_NAME, 0) + 1
            return output

        entries = calls.setdefault(TENSOR_IADD_FUNCTION_NAME, [])
        max_per_op = _profile_max_per_op()
        if not unique_capture and _profile_limit_reached(len(entries), max_per_op):
            key = f"{TENSOR_IADD_FUNCTION_NAME}:profile_max_per_op"
            skipped_counts[key] = skipped_counts.get(key, 0) + 1
            return output

        if lhs_before is None:
            lhs_before = _clone_for_profile(self)
        if other_before is None:
            other_before = _clone_for_profile(other)
        case_payload = None
        case_key = ""
        if unique_capture:
            case_payload = _profile_case_payload(
                TENSOR_IADD_FUNCTION_NAME,
                [],
                {"input": lhs_before, "other": other_before},
            )
            case_key, should_save = _record_unique_case(TENSOR_IADD_FUNCTION_NAME, case_payload)
            if not should_save:
                duplicate_key = f"{TENSOR_IADD_FUNCTION_NAME}:duplicate_profile_case"
                skipped_counts[duplicate_key] = skipped_counts.get(duplicate_key, 0) + 1
                return output

        entry = {
            "function_name": TENSOR_IADD_FUNCTION_NAME,
            "args": [],
            "kwargs": {
                "input": _serialize(lhs_before),
                "other": _serialize(other_before),
            },
            "output": _serialize(output),
            "signature": _TENSOR_IADD_SIGNATURE,
        }
        if unique_capture and case_payload is not None:
            _annotate_unique_entry(entry, case_key, case_payload)
        entries.append(entry)
        return output

    torch.Tensor.__iadd__ = wrapper


_PROFILE_MAX_ALL_VALUES = {"0", "all", "none", "unlimited"}


def _profile_limit_reached(count: int, max_per_op: int | None) -> bool:
    return max_per_op is not None and count >= max_per_op


def _format_profile_max_per_op(max_per_op: int | None) -> int | str:
    return "all" if max_per_op is None else max_per_op


def save_entries(
    func_name: str,
    entries: list[dict[str, Any]],
    base_dir: str,
    max_per_op: int | None = 200,
) -> None:
    func_dir = os.path.join(base_dir, _op_dir_name(func_name))
    os.makedirs(func_dir, exist_ok=True)

    existing_count = len(
        [
            n
            for n in os.listdir(func_dir)
            if n.startswith("entry_") and n.endswith(".pt")
        ]
    )
    if _profile_limit_reached(existing_count, max_per_op):
        return

    for idx, entry in enumerate(entries):
        if _profile_limit_reached(existing_count + idx, max_per_op):
            return
        file_name = f"entry_{existing_count + idx:06d}.pt"
        file_path = os.path.join(func_dir, file_name)
        torch.save(entry, file_path)
        case_key = entry.get("profile_case_key")
        if case_key:
            metadata = unique_case_metadata.setdefault(func_name, {})
            case_meta = metadata.setdefault(str(case_key), {"case_key": str(case_key)})
            case_meta["entry_file"] = file_name


def flush_calls(base_dir: str, max_per_op: int | None = 200) -> dict[str, int]:
    op_counts: dict[str, int] = {}
    for func_name, entries in calls.items():
        save_entries(func_name, entries, base_dir, max_per_op=max_per_op)
        op_counts[func_name] = op_counts.get(func_name, 0) + len(entries)
    calls.clear()
    return op_counts


def _profile_max_per_op() -> int | None:
    raw = os.environ.get("KFORGE_PROFILE_MAX_PER_OP", "").strip()
    if not raw:
        return 200
    if raw.lower() in _PROFILE_MAX_ALL_VALUES:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return 200
    if parsed <= 0:
        return None
    return max(parsed, 1)


def import_model_module(model_path: Path):
    with _patched_auto_docstring():
        spec = importlib.util.spec_from_file_location(model_path.stem, model_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module at {model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


def load_project_config(project_dir: Path) -> dict[str, Any]:
    config_path = project_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: failed to read config.json: {e}")
        return {}


def _call_with_optional_path(fn, path_val: str | None):
    if not path_val:
        return fn()
    try:
        sig = inspect.signature(fn)
        for name in [
            "data_dir",
            "dataset_path",
            "validation_path",
            "validation_set",
            "path",
            "root",
        ]:
            if name in sig.parameters:
                return fn(**{name: path_val})
        if len(sig.parameters) == 1:
            return fn(path_val)
    except Exception as e:
        print(f"Warning: failed to call dataloader with path: {e}")
    return fn()


def _call_with_optional_device(fn, weights_path: Path, device: str):
    try:
        sig = inspect.signature(fn)
        if "device" in sig.parameters:
            return fn(str(weights_path), device=device)
    except (ValueError, TypeError):
        pass
    return fn(str(weights_path))


def _instantiate_discovered_model(module):
    candidates = []
    for _, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        try:
            if not issubclass(obj, torch.nn.Module):
                continue
        except Exception:
            continue
        if obj.__module__ != module.__name__:
            continue
        candidates.append(obj)

    if not candidates:
        raise RuntimeError("Could not discover a model class in model.py")

    def _priority(cls):
        name = cls.__name__.lower()
        score = 0
        if "for" in name:
            score += 2
        if "model" in name:
            score += 1
        return score

    candidates.sort(key=_priority, reverse=True)

    last_error = None
    for cls in candidates:
        try:
            config_cls = getattr(cls, "config_class", None)
            if config_cls:
                try:
                    cfg = config_cls()
                    return cls(cfg)
                except Exception:
                    pass

            sig = inspect.signature(cls)
            required = []
            for p in sig.parameters.values():
                if p.name == "self":
                    continue
                if p.default == inspect.Parameter.empty:
                    required.append(p.name)
            if not required:
                return cls()
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise RuntimeError(f"Failed to instantiate discovered model class: {last_error}")
    raise RuntimeError("Failed to instantiate discovered model class")


def load_model(module, weights_path: Path, device: str):
    if hasattr(module, "load_weights") and weights_path.exists():
        return _call_with_optional_device(module.load_weights, weights_path, device)

    if hasattr(module, "build_model"):
        model = module.build_model()
    elif hasattr(module, "get_model"):
        model = module.get_model()
    else:
        model = _instantiate_discovered_model(module)

    if weights_path.exists():
        state = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(state, torch.nn.Module):
            model = state
        else:
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            try:
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"Warning: missing keys: {missing}, unexpected keys: {unexpected}")
            except Exception as e:
                print(f"Warning: failed to load state_dict: {e}")
    return model


def maybe_move_model_to_device(model, device: str):
    if getattr(model, "hf_device_map", None):
        return model
    return model.to(device)


def normalize_inputs(sample):
    if isinstance(sample, dict):
        return (), sample
    if isinstance(sample, (list, tuple)):
        if len(sample) == 2 and isinstance(sample[1], dict):
            args = sample[0] if isinstance(sample[0], (list, tuple)) else (sample[0],)
            return tuple(args), sample[1]
        return tuple(sample), {}
    return (sample,), {}


def move_to_device(obj, device: str):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def get_samples(module, max_batches: int, validation_path: str | None):
    data = None
    if validation_path and hasattr(module, "get_validation_dataloader"):
        data = _call_with_optional_path(module.get_validation_dataloader, validation_path)
    elif validation_path and hasattr(module, "get_dataloader"):
        data = _call_with_optional_path(module.get_dataloader, validation_path)
    elif hasattr(module, "sample_inputs"):
        data = module.sample_inputs()
    elif hasattr(module, "get_sample_inputs"):
        data = module.get_sample_inputs()
    elif hasattr(module, "make_example_input"):
        data = module.make_example_input()
    elif hasattr(module, "get_validation_dataloader"):
        data = _call_with_optional_path(module.get_validation_dataloader, validation_path)
    elif hasattr(module, "get_dataloader"):
        data = _call_with_optional_path(module.get_dataloader, validation_path)

    if isinstance(data, torch.utils.data.DataLoader):
        samples = []
        for i, batch in enumerate(data):
            if i >= max_batches:
                break
            samples.append(batch)
        return samples
    if isinstance(data, (list, tuple)):
        return list(data)[:max_batches]
    if data is not None:
        return [data] if max_batches > 0 else []
    return []


def _resolve_device() -> str:
    target = os.environ.get("KFORGE_TARGET_DEVICE", "").strip().lower()
    if target == "mps":
        if hasattr(torch, "backends") and torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("KFORGE_TARGET_DEVICE=mps but torch.backends.mps.is_available() returned False")
    if target in {"gpu", "cuda"}:
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("KFORGE_TARGET_DEVICE=cuda but torch.cuda.is_available() returned False")
    if target == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _collect_fallback_aten_stats(model, sample, device: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    try:
        from torch.profiler import ProfilerActivity

        activities = [ProfilerActivity.CPU]
        if device == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        args_tuple, kwargs = normalize_inputs(sample)
        args_tuple = move_to_device(args_tuple, device)
        kwargs = move_to_device(kwargs, device)

        with torch.no_grad():
            with torch.profiler.profile(activities=activities) as prof:
                try:
                    model(*args_tuple, **kwargs)
                except TypeError:
                    model(*args_tuple)

        for event in prof.key_averages():
            name = str(event.key) if hasattr(event, "key") else ""
            if not name.startswith("aten::"):
                continue
            op_name = name.replace("::", "_").replace(".", "_").replace("/", "_")
            count = int(getattr(event, "count", 0) or 0)
            if count <= 0:
                continue
            cpu_us = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
            avg_ms = (cpu_us / 1000.0) / float(count) if count > 0 else 0.0
            prev = out[op_name] if op_name in out else {"count": 0.0, "avg_ms": 0.0}
            total_count = float(prev["count"]) + float(count)
            weighted_ms = (float(prev["avg_ms"]) * float(prev["count"])) + (avg_ms * float(count))
            out[op_name] = {
                "count": total_count,
                "avg_ms": (weighted_ms / total_count) if total_count > 0 else 0.0,
            }
    except Exception:
        return {}
    return out


def _default_sample_from_model(model) -> Any:
    try:
        sig = inspect.signature(model.forward)
        params = sig.parameters
    except Exception:
        params = {}

    sample = {}
    if "pixel_values" in params:
        sample["pixel_values"] = torch.randn(1, 3, 224, 224)
    if "input_ids" in params:
        sample["input_ids"] = torch.randint(0, 1000, (1, 32), dtype=torch.long)
    if "attention_mask" in params:
        sample["attention_mask"] = torch.ones((1, 32), dtype=torch.long)

    if sample:
        return sample

    if "x" in params:
        return torch.randn(1, 3, 224, 224)
    return torch.randn(1, 3, 224, 224)


def main() -> int:
    global CAPTURE_ACTIVE
    parser = argparse.ArgumentParser(
        description="Profile a project model to capture per-op inputs/outputs."
    )
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--project-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    if args.project_dir:
        project_dir = Path(args.project_dir)
    elif args.project:
        project_dir = project_dir_for_name(args.project)
    else:
        raise RuntimeError("Provide --project or --project-dir")

    model_path = project_dir / "model.py"
    weights_path = project_dir / "weights.pt"
    if not model_path.exists():
        raise RuntimeError(f"Missing model.py at {model_path}")

    out_dir = Path(args.out_dir) if args.out_dir else project_dir / "io" / "individual_ops"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device()
    config = load_project_config(project_dir)
    _reset_profile_capture_state()
    _load_profile_filters(config)
    wrap_torch_ops()
    wrap_tensor_iadd()

    module = import_model_module(model_path)
    model = load_model(module, weights_path, device)
    maybe_move_model_to_device(model, device)
    model.eval()

    validation_raw = config.get("validation_dir") or config.get("validation_set") or ""
    validation_path = None
    if validation_raw:
        candidate = Path(validation_raw)
        if not candidate.is_absolute():
            candidate = project_dir / candidate
        if candidate.exists():
            validation_path = str(candidate)
        else:
            print(f"Warning: validation path not found: {candidate}")

    samples = get_samples(module, args.max_batches, validation_path)
    if not samples:
        samples = [_default_sample_from_model(model)]
    op_totals: dict[str, int] = {}
    op_profile_ms: dict[str, float] = {}
    max_per_op = _profile_max_per_op()
    capture_mode = _profile_capture_mode()

    with torch.no_grad():
        for sample in samples:
            args_tuple, kwargs = normalize_inputs(sample)
            args_tuple = move_to_device(args_tuple, device)
            kwargs = move_to_device(kwargs, device)
            CAPTURE_ACTIVE = True
            try:
                try:
                    model(*args_tuple, **kwargs)
                except TypeError:
                    model(*args_tuple)
            finally:
                CAPTURE_ACTIVE = False

            save_max_per_op = None if capture_mode == "unique" else max_per_op
            batch_counts = flush_calls(str(out_dir), max_per_op=save_max_per_op)
            for k, v in batch_counts.items():
                op_totals[k] = op_totals.get(k, 0) + v

    if capture_mode == "unique":
        unique_summary = _unique_case_summary()
        op_totals = {
            function_name: int(info.get("total_calls", 0))
            for function_name, info in unique_summary.get("ops", {}).items()
        }

    if samples and len(op_totals) < 3:
        fallback_stats = _collect_fallback_aten_stats(model, samples[0], device)
        for k in fallback_stats:
            info = fallback_stats[k]
            v = int(info["count"]) if "count" in info else 0
            if k not in op_totals:
                op_totals[k] = v
            if "avg_ms" in info:
                current_count = float(op_totals.get(k, 0))
                prev_ms = float(op_profile_ms.get(k, 0.0))
                prev_count = float(v) if v > 0 else current_count
                new_count = float(v)
                if k in op_profile_ms and prev_count > 0 and new_count > 0:
                    combined = (prev_ms * prev_count) + (float(info["avg_ms"]) * new_count)
                    op_profile_ms[k] = combined / (prev_count + new_count)
                else:
                    op_profile_ms[k] = float(info["avg_ms"])

    summary_path = out_dir.parent / "summary.json"
    unique_summary = _unique_case_summary() if capture_mode == "unique" else {}
    unique_cases_path = out_dir.parent / "unique_cases.json"
    summary = {
        "project": project_dir.name,
        "device": device,
        "profile_capture_mode": capture_mode,
        "profile_max_per_op": _format_profile_max_per_op(max_per_op),
        "op_counts": op_totals,
        "op_profile_ms": op_profile_ms,
        "skipped_counts": skipped_counts,
        "skip_filters": {
            "skip_ops": sorted(PROFILE_SKIP_OPS),
        },
    }
    if capture_mode == "unique":
        summary["op_unique_counts"] = {
            function_name: int(info.get("unique_cases", 0))
            for function_name, info in unique_summary.get("ops", {}).items()
        }
        summary["unique_cases_path"] = str(unique_cases_path)
        write_json_file(unique_cases_path, unique_summary)
    write_json_file(summary_path, summary)
    print(f"Saved profiling entries to {out_dir}")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
