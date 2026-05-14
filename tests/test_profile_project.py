from __future__ import annotations

import pytest
import torch

from src.optimizer.benchmarking import benchmark_ops
from src.optimizer.benchmarking import pipeline
from src.optimizer.benchmarking import profile_project
from src.optimizer.benchmarking.profile_project import get_samples


class _ListSampleModule:
    def sample_inputs(self):
        return ["a", "b", "c"]


class _ValidationPreferredModule:
    def __init__(self) -> None:
        self.sample_called = False
        self.validation_path = None

    def sample_inputs(self):
        self.sample_called = True
        return ["sample"]

    def get_validation_dataloader(self, validation_path=None):
        self.validation_path = validation_path
        return ["validation-a", "validation-b"]


def test_profile_project_limits_list_samples() -> None:
    assert get_samples(_ListSampleModule(), 2, None) == ["a", "b"]


def test_profile_project_prefers_validation_loader_when_path_is_configured() -> None:
    module = _ValidationPreferredModule()

    samples = get_samples(module, 1, "/tmp/validation")

    assert samples == ["validation-a"]
    assert module.validation_path == "/tmp/validation"
    assert module.sample_called is False


def test_profile_project_explicit_cuda_does_not_fallback_to_cpu(monkeypatch) -> None:
    monkeypatch.setenv("KFORGE_TARGET_DEVICE", "cuda")
    monkeypatch.setattr(profile_project.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="KFORGE_TARGET_DEVICE=cuda"):
        profile_project._resolve_device()


def test_benchmark_ops_explicit_cuda_does_not_fallback_to_cpu(monkeypatch) -> None:
    monkeypatch.setenv("KFORGE_TARGET_DEVICE", "cuda")
    monkeypatch.setattr(benchmark_ops.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="KFORGE_TARGET_DEVICE=cuda"):
        benchmark_ops._resolve_device()


def test_profile_project_accepts_unlimited_profile_entry_values(monkeypatch) -> None:
    for value in ("all", "unlimited", "none", "0"):
        monkeypatch.setenv("KFORGE_PROFILE_MAX_PER_OP", value)
        assert profile_project._profile_max_per_op() is None
        assert profile_project._format_profile_max_per_op(None) == "all"


def test_profile_project_keeps_default_and_integer_profile_limits(monkeypatch) -> None:
    monkeypatch.delenv("KFORGE_PROFILE_MAX_PER_OP", raising=False)
    assert profile_project._profile_max_per_op() == 200

    monkeypatch.setenv("KFORGE_PROFILE_MAX_PER_OP", "4")
    assert profile_project._profile_max_per_op() == 4


def test_profile_project_unique_case_signature_ignores_tensor_values() -> None:
    x1 = torch.zeros((2, 3), dtype=torch.float32)
    x2 = torch.ones((2, 3), dtype=torch.float32)

    first = profile_project._profile_case_payload(
        "torch.nn.functional.gelu",
        [],
        {"input": x1, "approximate": "none"},
    )
    second = profile_project._profile_case_payload(
        "torch.nn.functional.gelu",
        [],
        {"input": x2, "approximate": "none"},
    )

    assert profile_project._profile_case_key(first) == profile_project._profile_case_key(second)


def test_profile_project_unique_case_signature_includes_shape_and_kwargs() -> None:
    base = profile_project._profile_case_payload(
        "torch.nn.functional.conv2d",
        [],
        {
            "input": torch.zeros((1, 3, 16, 16)),
            "weight": torch.zeros((8, 3, 3, 3)),
            "bias": None,
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
        },
    )
    changed_shape = profile_project._profile_case_payload(
        "torch.nn.functional.conv2d",
        [],
        {
            "input": torch.zeros((1, 3, 32, 32)),
            "weight": torch.zeros((8, 3, 3, 3)),
            "bias": None,
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
        },
    )
    changed_stride = profile_project._profile_case_payload(
        "torch.nn.functional.conv2d",
        [],
        {
            "input": torch.zeros((1, 3, 16, 16)),
            "weight": torch.zeros((8, 3, 3, 3)),
            "bias": None,
            "stride": (2, 2),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
        },
    )

    assert profile_project._profile_case_key(base) != profile_project._profile_case_key(changed_shape)
    assert profile_project._profile_case_key(base) != profile_project._profile_case_key(changed_stride)


def test_profile_project_records_unique_case_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KFORGE_PROFILE_CAPTURE_MODE", "unique")
    profile_project._reset_profile_capture_state()
    func_name = "torch.nn.functional.gelu"
    payload = profile_project._profile_case_payload(
        func_name,
        [],
        {"input": torch.zeros((2, 3)), "approximate": "none"},
    )

    case_key, is_new = profile_project._record_unique_case(func_name, payload)
    duplicate_key, duplicate_is_new = profile_project._record_unique_case(func_name, payload)
    assert is_new is True
    assert duplicate_key == case_key
    assert duplicate_is_new is False

    entry = profile_project._annotate_unique_entry(
        {
            "function_name": func_name,
            "args": [],
            "kwargs": {"input": torch.zeros((2, 3)), "approximate": "none"},
            "output": torch.zeros((2, 3)),
        },
        case_key,
        payload,
    )
    profile_project.calls[func_name] = [entry]
    profile_project.flush_calls(str(tmp_path), max_per_op=None)
    summary = profile_project._unique_case_summary()

    op_summary = summary["ops"][func_name]
    assert op_summary["total_calls"] == 2
    assert op_summary["unique_cases"] == 1
    assert op_summary["cases"][0]["entry_file"] == "entry_000000.pt"
    assert (tmp_path / "torch_nn_functional_gelu" / "entry_000000.pt").exists()


def test_benchmark_ops_accepts_all_max_entries(tmp_path) -> None:
    for name in ("entry_000002.pt", "entry_000001.pt", "entry_000003.pt"):
        (tmp_path / name).write_text("placeholder", encoding="utf-8")

    assert benchmark_ops._parse_max_entries("all") is None
    assert [path.name for path in benchmark_ops._selected_entry_files(tmp_path, None)] == [
        "entry_000001.pt",
        "entry_000002.pt",
        "entry_000003.pt",
    ]
    assert [path.name for path in benchmark_ops._selected_entry_files(tmp_path, 2)] == [
        "entry_000001.pt",
        "entry_000002.pt",
    ]


def test_pipeline_forwards_all_profile_limit_to_benchmark(monkeypatch) -> None:
    monkeypatch.setenv("KFORGE_PROFILE_MAX_PER_OP", "all")
    monkeypatch.delenv("KFORGE_PROFILE_CAPTURE_MODE", raising=False)
    assert pipeline._benchmark_max_entries_args_from_env() == ["--max-entries", "all"]

    monkeypatch.setenv("KFORGE_PROFILE_MAX_PER_OP", "4")
    assert pipeline._benchmark_max_entries_args_from_env() == []

    monkeypatch.setenv("KFORGE_PROFILE_CAPTURE_MODE", "unique")
    assert pipeline._benchmark_max_entries_args_from_env() == ["--max-entries", "all"]
