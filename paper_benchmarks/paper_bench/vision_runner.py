from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .artifacts import RunLayout, write_json_artifact
from .baselines import CompileSettings, compile_model, compile_settings_from_dict, resolve_torch_dtype, sync_device
from .correctness import reference_correctness
from .kf_runtime import build_kf_runtime_details, get_cast_runtime_stats, load_cast_model, reset_cast_runtime_stats, runtime_settings_from_dict
from .provenance import sha256_file
from .schema import BenchmarkArtifact, BenchmarkMode, CorrectnessStatus, EnvironmentArtifact, RunManifestArtifact, Stage, Variant
from .stats import build_latency_summary
from .validator import validated_artifact_update


@dataclass(frozen=True)
class ImageRecord:
    record_id: str
    path: Path
    label: int | None
    sha256: str | None


def _artifact_common_fields(common_fields: dict[str, Any]) -> dict[str, Any]:
    artifact_common = dict(common_fields)
    for key in (
        "artifact_type",
        "benchmark_mode",
        "variant",
        "stage",
        "warmup_count",
        "timed_run_count",
        "latency_samples_ms",
        "latency_summary",
        "sample_records",
        "correctness_status",
        "correctness_message",
        "fallback_count",
        "kernel_hit_count",
        "compile_time_ms",
        "steady_state_time_ms",
        "prompt_id",
        "prompt_hash",
        "token_count",
        "configured_batch_size",
        "comparison_group",
        "prompt_bucket_id",
        "details",
    ):
        artifact_common.pop(key, None)
    return artifact_common


def _import_model_module(model_path: str | Path):
    path = Path(model_path)
    if path.is_dir():
        path = path / "model.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"kforge_vision_model_{hash(path)}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_project_vision_model(model_spec, device: str | None = None) -> tuple[Any, None, float]:
    module = _import_model_module(model_spec.model_path)
    started = time.perf_counter()
    if hasattr(module, "build_model"):
        model = module.build_model()
    elif hasattr(module, "get_model"):
        model = module.get_model()
    else:
        raise RuntimeError(f"{model_spec.model_path} does not define build_model()")
    model.eval()
    dtype = resolve_torch_dtype(getattr(model_spec, "torch_dtype", None))
    if device:
        if dtype is not None and str(device).startswith("cuda"):
            model.to(device=device, dtype=dtype)
        else:
            model.to(device=device)
    load_ms = (time.perf_counter() - started) * 1000.0
    return model, None, load_ms


def load_image_records(workload_path: str | Path, *, max_images: int | None = None) -> list[ImageRecord]:
    path = Path(workload_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = path / "manifest.jsonl" if path.is_dir() else path
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    records: list[ImageRecord] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        raw_path = row.get("path") or row.get("image_path") or row.get("file")
        if not raw_path:
            continue
        image_path = Path(str(raw_path))
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        label = row.get("label")
        records.append(
            ImageRecord(
                record_id=str(row.get("id") or image_path.stem),
                path=image_path,
                label=int(label) if label is not None else None,
                sha256=str(row.get("sha256")) if row.get("sha256") else None,
            )
        )
        if max_images is not None and len(records) >= max_images:
            break
    if not records:
        raise ValueError(f"No image records found in {manifest}")
    missing = [str(record.path) for record in records if not record.path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image files in manifest: {missing[:5]}")
    return records


def _batch_records(records: list[ImageRecord], batch_size: int) -> list[list[ImageRecord]]:
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def _load_batch(records: list[ImageRecord], *, device: str, dtype: torch.dtype | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    from PIL import Image
    from torchvision.models import ResNet50_Weights

    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    images = []
    labels = []
    for record in records:
        with Image.open(record.path) as image:
            images.append(transform(image.convert("RGB")))
        if record.label is not None:
            labels.append(record.label)
    batch = torch.stack(images, dim=0)
    target_dtype = dtype if dtype is not None and str(device).startswith("cuda") else None
    batch = batch.to(device=device, dtype=target_dtype, non_blocking=False)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device) if len(labels) == len(records) else None
    return batch, label_tensor


def _hash_tensor(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _timed_forward(model: Any, batch: torch.Tensor, *, device: str) -> tuple[float, torch.Tensor]:
    sync_device(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(batch)
    sync_device(device)
    return (time.perf_counter() - started) * 1000.0, logits


def _topk_counts(logits: torch.Tensor, labels: torch.Tensor | None) -> tuple[int, int, int]:
    if labels is None:
        return 0, 0, int(logits.shape[0])
    _, pred = logits.topk(5, dim=1)
    correct = pred.eq(labels.view(-1, 1))
    top1 = int(correct[:, :1].any(dim=1).sum().item())
    top5 = int(correct.any(dim=1).sum().item())
    return top1, top5, int(labels.numel())


def _compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[CorrectnessStatus, str | None, dict[str, Any]]:
    if reference.shape != candidate.shape:
        return CorrectnessStatus.failed, f"shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}", {}
    ref_top1 = reference.argmax(dim=1)
    cand_top1 = candidate.argmax(dim=1)
    top1_matches = int(ref_top1.eq(cand_top1).sum().item())
    total = int(ref_top1.numel())
    ref_f32 = reference.detach().float()
    cand_f32 = candidate.detach().float()
    max_abs = float((ref_f32 - cand_f32).abs().max().item())
    allclose = bool(torch.allclose(ref_f32, cand_f32, atol=5e-2, rtol=5e-2))
    details = {
        "top1_matches_eager": top1_matches,
        "top1_total": total,
        "max_abs_diff_vs_eager": max_abs,
        "logits_allclose_atol": 5e-2,
        "logits_allclose_rtol": 5e-2,
        "logits_allclose": allclose,
    }
    if top1_matches != total:
        return CorrectnessStatus.failed, f"top1 mismatch against eager: {top1_matches}/{total}", details
    if not allclose:
        return CorrectnessStatus.failed, f"logits mismatch against eager; max_abs_diff={max_abs:.6g}", details
    return CorrectnessStatus.passed, None, details


def _metric_artifact_path(layout: RunLayout, variant: Variant, stage: Stage, batch_size: int) -> Path:
    return layout.metrics_dir / f"{variant.value}_{stage.value}_bs{batch_size}.json"


def _write_stage_artifact(
    layout: RunLayout,
    common_fields: dict[str, Any],
    *,
    variant: Variant,
    stage: Stage,
    batch_size: int,
    samples_ms: list[float],
    warmup_count: int,
    timed_run_count: int,
    correctness_status: CorrectnessStatus,
    correctness_message: str | None,
    compile_time_ms: float | None,
    steady_state_time_ms: float | None,
    fallback_count: int | None,
    kernel_hit_count: int | None,
    details: dict[str, Any],
    sample_records: list[dict[str, Any]] | None = None,
) -> Path:
    artifact_common = _artifact_common_fields(common_fields)
    artifact = BenchmarkArtifact(
        **artifact_common,
        artifact_type="benchmark_result",
        benchmark_mode=BenchmarkMode.e2e_model,
        variant=variant,
        stage=stage,
        warmup_count=warmup_count,
        timed_run_count=timed_run_count,
        latency_samples_ms=samples_ms,
        latency_summary=build_latency_summary(samples_ms),
        sample_records=sample_records or [],
        correctness_status=correctness_status,
        correctness_message=correctness_message,
        steady_state_time_ms=steady_state_time_ms,
        compile_time_ms=compile_time_ms,
        fallback_count=fallback_count,
        kernel_hit_count=kernel_hit_count,
        configured_batch_size=batch_size,
        comparison_group=f"image_224_bs{batch_size}",
        prompt_bucket_id="image_224",
        details=details,
    )
    artifact = validated_artifact_update(artifact)
    return write_json_artifact(_metric_artifact_path(layout, variant, stage, batch_size), artifact)


def _variant_common(common_fields: dict[str, Any], variant: Variant) -> dict[str, Any]:
    common = dict(common_fields)
    if variant != Variant.kf_cast:
        common["cast_package_path"] = None
        common["cast_package_hash"] = None
        common["kf_artifact_path"] = None
        common["kf_artifact_hash"] = None
        common["kf_artifact_kind"] = None
    return common


def _run_batches(
    *,
    candidate_model: Any,
    reference_model: Any | None,
    records: list[ImageRecord],
    batch_size: int,
    device: str,
    dtype: torch.dtype | None,
    variant: Variant,
    timed_batch_limit: int,
) -> tuple[list[float], list[dict[str, Any]], CorrectnessStatus, str | None, dict[str, Any]]:
    samples: list[float] = []
    sample_records: list[dict[str, Any]] = []
    failures: list[str] = []
    correct_top1 = 0
    correct_top5 = 0
    total_images = 0
    eager_match_count = 0
    eager_checked_count = 0
    max_abs_diff = 0.0
    batches = _batch_records(records, batch_size)[:timed_batch_limit]
    for batch_index, batch_records in enumerate(batches):
        batch, labels = _load_batch(batch_records, device=device, dtype=dtype)
        reference_logits = None
        reference_hash = None
        if reference_model is not None:
            with torch.inference_mode():
                reference_logits = reference_model(batch)
            reference_hash = _hash_tensor(reference_logits)
        elapsed_ms, logits = _timed_forward(candidate_model, batch, device=device)
        output_hash = _hash_tensor(logits)
        top1, top5, n_images = _topk_counts(logits, labels)
        correct_top1 += top1
        correct_top5 += top5
        total_images += n_images
        status = CorrectnessStatus.reference if variant == Variant.eager else CorrectnessStatus.passed
        message = None
        compare_details: dict[str, Any] = {}
        if reference_logits is not None:
            status, message, compare_details = _compare_logits(reference_logits, logits)
            eager_match_count += int(compare_details.get("top1_matches_eager", 0))
            eager_checked_count += int(compare_details.get("top1_total", 0))
            max_abs_diff = max(max_abs_diff, float(compare_details.get("max_abs_diff_vs_eager", 0.0) or 0.0))
            if status != CorrectnessStatus.passed:
                failures.append(message or f"batch {batch_index} failed correctness")
        samples.append(float(elapsed_ms))
        sample_records.append(
            {
                "sample_index": batch_index,
                "record_ids": [record.record_id for record in batch_records],
                "image_paths": [str(record.path) for record in batch_records],
                "image_hashes": [record.sha256 or sha256_file(record.path) for record in batch_records],
                "labels": [record.label for record in batch_records],
                "batch_size": len(batch_records),
                "latency_ms": float(elapsed_ms),
                "images_per_second": (len(batch_records) * 1000.0 / float(elapsed_ms)) if elapsed_ms > 0 else None,
                "top1_correct": top1,
                "top5_correct": top5,
                "output_hash": output_hash,
                "output_token_hashes": [output_hash],
                "reference_output_hash": reference_hash,
                "correctness_status": status.value,
                "correctness_message": message,
                "logit_comparison": compare_details,
            }
        )
    if variant == Variant.eager:
        correctness_status, correctness_message = reference_correctness()
    else:
        correctness_status = CorrectnessStatus.passed if not failures else CorrectnessStatus.failed
        correctness_message = "; ".join(failures[:8]) if failures else None
    aggregate = {
        "image_count": total_images,
        "top1_correct": correct_top1,
        "top5_correct": correct_top5,
        "top1_accuracy": (correct_top1 / total_images) if total_images else None,
        "top5_accuracy": (correct_top5 / total_images) if total_images else None,
        "eager_top1_match_count": eager_match_count,
        "eager_top1_checked_count": eager_checked_count,
        "max_abs_diff_vs_eager": max_abs_diff,
        "correctness_checked_run_count": len(samples),
        "per_run_output_hash_verification": variant != Variant.eager,
        "total_images_per_second": (total_images * 1000.0 / sum(samples)) if samples else None,
    }
    return samples, sample_records, correctness_status, correctness_message, aggregate


def run_vision_benchmark(
    *,
    layout: RunLayout,
    common_fields: dict[str, Any],
    env_artifact: EnvironmentArtifact,
    manifest_artifact: RunManifestArtifact,
    model_spec,
    suite,
    variant: Variant,
    model_loader: Callable[..., tuple[Any, Any, float]] = load_project_vision_model,
    compile_model_fn: Callable[[Any, CompileSettings | dict[str, Any] | None], tuple[Any, float]] = compile_model,
    cast_loader: Callable[..., tuple[Any, dict[str, Any]]] = load_cast_model,
    max_images: int | None = None,
    full_workload: bool = False,
) -> RunLayout:
    write_json_artifact(layout.run_dir / "manifest.json", manifest_artifact)
    write_json_artifact(layout.run_dir / "env.json", env_artifact)

    device = getattr(suite, "device", None) or getattr(model_spec, "device", None) or "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ResNet-50 vision benchmark requested CUDA but torch.cuda.is_available() is false")
    dtype = resolve_torch_dtype(getattr(model_spec, "torch_dtype", None))
    records = load_image_records(suite.workload_path, max_images=max_images)
    compile_settings = compile_settings_from_dict(common_fields.get("compile_settings"))
    kf_settings = runtime_settings_from_dict(common_fields.get("kf_settings"))
    common = _variant_common(
        {
            **common_fields,
            "compile_settings": compile_settings.as_dict(),
            "kf_settings": kf_settings.as_dict(),
        },
        variant,
    )

    if variant == Variant.kf_cast:
        if not kf_settings.cast_package_path:
            raise ValueError("kf_cast vision benchmarking requires --cast-package or model cast_package_path")
        candidate_model, runtime_meta = cast_loader(kf_settings.cast_package_path, device=device, settings=kf_settings)
        if dtype is not None and str(device).startswith("cuda"):
            candidate_model.to(device=device, dtype=dtype)
        elif device:
            candidate_model.to(device=device)
        load_time_ms = float(runtime_meta.get("load_time_ms", runtime_meta.get("runtime_load_time_ms", 0.0)) or 0.0)
        fallback_count = int(runtime_meta.get("fallback_count", 0) or 0)
        kernel_hit_count = int(runtime_meta.get("kernel_hit_count", 0) or 0)
        runtime_details = build_kf_runtime_details(runtime_meta)
    else:
        candidate_model, _, load_time_ms = model_loader(model_spec, device=device)
        runtime_meta = None
        fallback_count = None
        kernel_hit_count = None
        runtime_details = {}

    reference_model = None
    if variant != Variant.eager:
        reference_model, _, _ = model_loader(model_spec, device=device)

    compile_time_ms: float | None = None
    if variant == Variant.torch_compile:
        candidate_model, compile_time_ms = compile_model_fn(candidate_model, compile_settings)

    for batch_size in list(suite.batch_sizes or [suite.batch_size]):
        _write_stage_artifact(
            layout,
            common,
            variant=variant,
            stage=Stage.load,
            batch_size=int(batch_size),
            samples_ms=[float(load_time_ms)],
            warmup_count=0,
            timed_run_count=1,
            correctness_status=CorrectnessStatus.not_applicable,
            correctness_message=None,
            compile_time_ms=None,
            steady_state_time_ms=float(load_time_ms),
            fallback_count=fallback_count,
            kernel_hit_count=kernel_hit_count,
            details={
                "load_source": "cast_package" if variant == Variant.kf_cast else "torchvision_project_model",
                "workload_type": "image_classification",
                **runtime_details,
            },
            sample_records=[{"sample_index": 0, "latency_ms": float(load_time_ms)}],
        )
        if compile_time_ms is not None:
            _write_stage_artifact(
                layout,
                common,
                variant=variant,
                stage=Stage.compile,
                batch_size=int(batch_size),
                samples_ms=[float(compile_time_ms)],
                warmup_count=0,
                timed_run_count=1,
                correctness_status=CorrectnessStatus.not_applicable,
                correctness_message=None,
                compile_time_ms=float(compile_time_ms),
                steady_state_time_ms=None,
                fallback_count=fallback_count,
                kernel_hit_count=kernel_hit_count,
                details={"compile_settings": compile_settings.as_dict(), "workload_type": "image_classification"},
                sample_records=[{"sample_index": 0, "latency_ms": float(compile_time_ms)}],
            )

        warmup_batches = _batch_records(records, int(batch_size))[: int(suite.warmup_count)]
        warmup_samples: list[float] = []
        warmup_records: list[dict[str, Any]] = []
        for warmup_index, batch_records in enumerate(warmup_batches):
            batch, _ = _load_batch(batch_records, device=device, dtype=dtype)
            elapsed_ms, logits = _timed_forward(candidate_model, batch, device=device)
            warmup_samples.append(float(elapsed_ms))
            warmup_records.append(
                {
                    "sample_index": warmup_index,
                    "record_ids": [record.record_id for record in batch_records],
                    "batch_size": len(batch_records),
                    "latency_ms": float(elapsed_ms),
                    "output_hash": _hash_tensor(logits),
                }
            )
        if warmup_samples:
            _write_stage_artifact(
                layout,
                common,
                variant=variant,
                stage=Stage.warmup,
                batch_size=int(batch_size),
                samples_ms=warmup_samples,
                warmup_count=int(suite.warmup_count),
                timed_run_count=len(warmup_samples),
                correctness_status=CorrectnessStatus.not_applicable,
                correctness_message=None,
                compile_time_ms=compile_time_ms,
                steady_state_time_ms=sum(warmup_samples) / len(warmup_samples),
                fallback_count=fallback_count,
                kernel_hit_count=kernel_hit_count,
                details={"workload_type": "image_classification", **runtime_details},
                sample_records=warmup_records,
            )

        if variant == Variant.kf_cast:
            reset_cast_runtime_stats(candidate_model)
        total_batches = len(_batch_records(records, int(batch_size)))
        timed_batch_limit = total_batches if full_workload else min(total_batches, int(suite.timed_run_count))
        samples, sample_records, correctness_status, correctness_message, aggregate = _run_batches(
            candidate_model=candidate_model,
            reference_model=reference_model,
            records=records,
            batch_size=int(batch_size),
            device=device,
            dtype=dtype,
            variant=variant,
            timed_batch_limit=timed_batch_limit,
        )
        if variant == Variant.kf_cast:
            stats = get_cast_runtime_stats(candidate_model)
            fallback_count = int(stats.get("fallbacks_to_original", fallback_count or 0) or 0)
            kernel_hit_count = int(stats.get("kernel_launches_succeeded", kernel_hit_count or 0) or 0)
            runtime_details = build_kf_runtime_details(runtime_meta or {}, stats)
        details = {
            "workload_type": "image_classification",
            "dataset_record_count": len(records),
            "full_workload": bool(full_workload),
            "timed_batch_limit": timed_batch_limit,
            "batch_size": int(batch_size),
            "image_size": 224,
            "preprocessing": "torchvision ResNet50_Weights.IMAGENET1K_V2.transforms()",
            "selection_policy": {"method": "manifest_order", "post_hoc": False},
            **aggregate,
            **runtime_details,
        }
        (layout.raw_dir / f"{variant.value}_vision_bs{batch_size}.json").write_text(
            json.dumps(sample_records, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_stage_artifact(
            layout,
            common,
            variant=variant,
            stage=Stage.total_generate,
            batch_size=int(batch_size),
            samples_ms=samples,
            warmup_count=int(suite.warmup_count),
            timed_run_count=len(samples),
            correctness_status=correctness_status,
            correctness_message=correctness_message,
            compile_time_ms=compile_time_ms,
            steady_state_time_ms=(sum(samples) / len(samples)) if samples else 0.0,
            fallback_count=fallback_count,
            kernel_hit_count=kernel_hit_count,
            details=details,
            sample_records=sample_records,
        )
    return layout
