from __future__ import annotations

import pytest

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
    assert pipeline._benchmark_max_entries_args_from_env() == ["--max-entries", "all"]

    monkeypatch.setenv("KFORGE_PROFILE_MAX_PER_OP", "4")
    assert pipeline._benchmark_max_entries_args_from_env() == []
