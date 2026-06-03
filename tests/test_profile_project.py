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


@pytest.fixture(autouse=True)
def _reset_profile_filters():
    profile_project._load_profile_filters({})
    profile_project._reset_profile_capture_state()
    profile_project.CAPTURE_ACTIVE = False
    yield
    profile_project.CAPTURE_ACTIVE = False
    profile_project._reset_profile_capture_state()
    profile_project._restore_tensor_method_wrappers()
    profile_project._load_profile_filters({})


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


@pytest.mark.parametrize(
    "full_key",
    [
        "torch.reshape",
        "torch.Tensor.view",
        "torch.randn",
        "torch.randint",
        "torch.flatten",
        "torch.to",
        "torch.clone",
        "torch.empty",
        "torch.nn.functional.linear",
    ],
)
def test_profile_project_profiles_non_dropout_ops_by_default(full_key: str) -> None:
    assert profile_project._should_skip(full_key) is False


@pytest.mark.parametrize(
    "full_key",
    [
        "torch.nn.functional.dropout",
        "torch.nn.functional.dropout_",
        "torch.nn.functional.alpha_dropout",
        "torch.nn.functional.feature_alpha_dropout",
    ],
)
def test_profile_project_skips_dropout_ops(full_key: str) -> None:
    assert profile_project._should_skip(full_key) is True


def test_profile_project_ignores_config_allowlist_and_skiplist() -> None:
    profile_project._load_profile_filters(
        {
            "profile": {
                "allow_ops": ["linear"],
                "allowlist": ["conv2d"],
                "skip_ops": ["reshape"],
                "skiplist": ["flatten"],
                "skip_prefixes": ["rand"],
            }
        }
    )

    assert profile_project._should_skip("torch.nn.functional.gelu") is False
    assert profile_project._should_skip("torch.reshape") is False
    assert profile_project._should_skip("torch.randn") is False
    assert profile_project._should_skip("torch.nn.functional.dropout") is True


def test_profile_project_wraps_torch_callables_but_not_classes(monkeypatch) -> None:
    def compute_op(value):
        return value

    def foreign_op(value):
        return value

    compute_op.__module__ = "torch"
    foreign_op.__module__ = "not_torch"

    fake_module = type(
        "FakeTorchModule",
        (),
        {
            "__name__": "torch",
            "Tensor": type("Tensor", (), {}),
            "compute_op": compute_op,
            "foreign_op": foreign_op,
        },
    )()
    wrapped: list[str] = []
    monkeypatch.setattr(
        profile_project,
        "wrap_function",
        lambda module, func_name: wrapped.append(func_name),
    )

    profile_project._wrap_module_callables(fake_module)

    assert wrapped == ["compute_op"]


def test_profile_project_records_bound_tensor_methods() -> None:
    x = torch.arange(12)
    profile_project.wrap_tensor_methods()

    profile_project.CAPTURE_ACTIVE = True
    try:
        viewed = x.view(3, 4)
        reshaped = x.reshape(2, 6)
        flattened = viewed.flatten()
        converted = flattened.to(dtype=torch.float64)
    finally:
        profile_project.CAPTURE_ACTIVE = False

    assert reshaped.shape == (2, 6)
    assert converted.dtype == torch.float64
    for key in [
        "torch.tensor.view",
        "torch.tensor.reshape",
        "torch.tensor.flatten",
        "torch.tensor.to",
    ]:
        assert key in profile_project.calls
        assert len(profile_project.calls[key]) == 1
        assert profile_project.calls[key][0]["function_name"] == key


def test_profile_project_tensor_methods_respect_capture_gate() -> None:
    x = torch.arange(6)
    profile_project.wrap_tensor_methods()

    assert x.view(2, 3).shape == (2, 3)
    assert profile_project.calls == {}


def test_generator_monitor_replays_bound_tensor_methods() -> None:
    from src.generator import main as generator_main

    cases = [
        ("torch.tensor.view", [torch.arange(6), 2, 3], {}, (2, 3), None),
        ("torch.tensor.reshape", [torch.arange(6), 3, 2], {}, (3, 2), None),
        ("torch.tensor.flatten", [torch.arange(6).view(2, 3)], {}, (6,), None),
        ("torch.tensor.to", [torch.arange(6)], {"dtype": torch.float64}, (6,), torch.float64),
    ]

    for function_name, args, kwargs, expected_shape, expected_dtype in cases:
        expr = generator_main._monitor_exec_for_function(function_name)
        output = eval(expr, {"torch": torch}, {"args": args, "kwargs": kwargs})

        assert tuple(output.shape) == expected_shape
        if expected_dtype is not None:
            assert output.dtype == expected_dtype


def test_benchmark_ops_resolves_bound_tensor_method_baselines() -> None:
    x = torch.arange(6)

    view = benchmark_ops._get_pytorch_func("torch_tensor_view")
    reshape = benchmark_ops._get_pytorch_func("torch_tensor_reshape")
    flatten = benchmark_ops._get_pytorch_func("torch_tensor_flatten")
    to_dtype = benchmark_ops._get_pytorch_func("torch_tensor_to")

    assert view is not None
    assert reshape is not None
    assert flatten is not None
    assert to_dtype is not None
    assert view(x, 2, 3).shape == (2, 3)
    assert reshape(x, 3, 2).shape == (3, 2)
    assert flatten(x.view(2, 3)).shape == (6,)
    assert to_dtype(x, dtype=torch.float64).dtype == torch.float64


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


def test_profile_project_records_allowlisted_torch_flatten(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KFORGE_PROFILE_CAPTURE_MODE", "unique")
    monkeypatch.setenv("KFORGE_PROFILE_MAX_PER_OP", "all")
    profile_project._reset_profile_capture_state()
    profile_project._load_profile_filters({"profile": {"allow_ops": ["flatten"]}})

    try:
        profile_project.wrap_torch_top_level_functions()

        x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        profile_project.CAPTURE_ACTIVE = True
        try:
            y = torch.flatten(x, 1)
        finally:
            profile_project.CAPTURE_ACTIVE = False

        assert y.shape == (2, 12)
        assert "torch.flatten" in profile_project.calls

        profile_project.flush_calls(str(tmp_path), max_per_op=None)
        summary = profile_project._unique_case_summary()
        op_summary = summary["ops"]["torch.flatten"]

        assert op_summary["op_dir"] == "torch_flatten"
        assert op_summary["total_calls"] == 1
        assert op_summary["unique_cases"] == 1

        entry_path = tmp_path / "torch_flatten" / "entry_000000.pt"
        assert entry_path.exists()
        entry = torch.load(entry_path, map_location="cpu", weights_only=False)
        assert entry["function_name"] == "torch.flatten"
        assert entry["signature"]["params"] == ["input", "start_dim", "end_dim"]
        assert entry["kwargs"]["start_dim"] == 1
        assert entry["kwargs"]["end_dim"] == -1
        assert torch.equal(entry["output"], y)
    finally:
        profile_project.CAPTURE_ACTIVE = False
        profile_project._reset_profile_capture_state()
        profile_project._load_profile_filters({})


def test_benchmark_ops_maps_torch_flatten() -> None:
    assert benchmark_ops._get_pytorch_func("torch_flatten") is torch.flatten


def test_benchmark_ops_maps_sd35_missing_ops() -> None:
    assert benchmark_ops._get_pytorch_func("torch_nn_functional_embedding") is torch.nn.functional.embedding
    assert benchmark_ops._get_pytorch_func("torch_nn_functional_softmax") is torch.nn.functional.softmax
    assert benchmark_ops._get_pytorch_func("torch_nn_functional_pad") is torch.nn.functional.pad
    assert benchmark_ops._get_pytorch_func("torch_nn_functional_interpolate") is torch.nn.functional.interpolate
    assert benchmark_ops._get_pytorch_func("torch_tensor_iadd") is torch.add


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
