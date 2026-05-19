from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "paper_benchmarks" / "runs" / "trace_weighted_unique_replay"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_path(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry_files_hash(entry_files: list[Path]) -> str:
    digest_payload = [
        {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in entry_files
    ]
    return hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode("utf-8")).hexdigest()


def latest_manifest(project: str, arm: str) -> Path | None:
    root = REPO_ROOT / "paper_benchmarks" / "data_collection" / "artifacts" / project
    manifests = sorted(root.glob(f"*__{arm}__*/collection_manifest.json"), key=lambda p: p.stat().st_mtime)
    return manifests[-1] if manifests else None


def selected_kernel_map(project_dir: Path, arm: str, *, prefer_manifest: bool = True) -> dict[str, str]:
    manifest = latest_manifest(project_dir.name, arm)
    if prefer_manifest and manifest is not None:
        payload = read_json(manifest, {})
        selected = payload.get("selected_kernel_map_for_arm")
        if isinstance(selected, dict) and selected:
            return {str(k): str(v) for k, v in selected.items()}

    from paper_benchmarks.data_collection.collect_zero_shot import (
        selected_tree_kernel_map,
        successful_zero_shot_kernel_map,
    )

    if arm == "zero_shot":
        return successful_zero_shot_kernel_map(project_dir)
    selected, _details = selected_tree_kernel_map(project_dir, arm=arm)
    return selected


def discovered_ops(project_dir: Path) -> list[str]:
    io_root = project_dir / "io" / "individual_ops"
    if not io_root.exists():
        return []
    return sorted(child.name for child in io_root.iterdir() if child.is_dir())


def weighted_ms(measurement: dict[str, Any], counts_by_entry: dict[str, int]) -> dict[str, Any]:
    rows = measurement.get("entry_results")
    if not isinstance(rows, list):
        rows = [
            {"entry_file": entry_file, "latency_ms": latency}
            for entry_file, latency in zip(
                measurement.get("entry_files") or [],
                measurement.get("entry_latencies_ms") or [],
            )
        ]

    latency_by_entry: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        entry_file = Path(str(row.get("entry_file") or "")).name
        if not entry_file:
            continue
        try:
            latency_by_entry[entry_file] = float(row.get("latency_ms"))
        except Exception:
            continue

    weighted_total = 0.0
    measured_weight = 0
    expected_weight = 0
    missing: list[str] = []
    for entry_file, raw_count in sorted(counts_by_entry.items()):
        count = int(raw_count)
        if count <= 0:
            continue
        expected_weight += count
        if entry_file not in latency_by_entry:
            missing.append(entry_file)
            continue
        weighted_total += latency_by_entry[entry_file] * float(count)
        measured_weight += count

    avg_ms = weighted_total / float(measured_weight) if measured_weight > 0 else None
    return {
        "avg_ms": avg_ms,
        "weighted_total_ms": weighted_total,
        "expected_weight": expected_weight,
        "measured_weight": measured_weight,
        "missing_entries": missing,
        "complete": measured_weight == expected_weight and not missing,
        "latency_by_entry": latency_by_entry,
    }


def measure_torch_op(
    *,
    cache_root: Path,
    project: str,
    op: str,
    entry_files: list[Path],
    device: str,
    force: bool,
) -> dict[str, Any]:
    from src.optimizer.benchmarking.benchmark_ops import _get_pytorch_func, _measure_pytorch_files

    cache_path = cache_root / "cache" / "torch" / f"{op}.json"
    entries_hash = entry_files_hash(entry_files)
    cached = read_json(cache_path, {})
    if (
        not force
        and isinstance(cached, dict)
        and cached.get("entries_hash") == entries_hash
        and cached.get("entry_count") == len(entry_files)
        and cached.get("project") == project
    ):
        return cached["measurement"]

    func = _get_pytorch_func(op)
    if func is None:
        raise RuntimeError(f"No PyTorch replay function for {op}")
    measurement = _measure_pytorch_files(func, entry_files, device)
    write_json(
        cache_path,
        {
            "project": project,
            "op": op,
            "entry_count": len(entry_files),
            "entries_hash": entries_hash,
            "measurement": measurement,
            "created_at_utc": utc_now(),
        },
    )
    return measurement


def profile_kernel_source(
    *,
    cache_root: Path,
    project: str,
    arm: str,
    op: str,
    source: Path,
    io_dir: Path,
    entry_files: list[Path],
    force: bool,
) -> dict[str, Any]:
    from src.optimizer.backends.cuda import CUDABackend

    cache_path = cache_root / "cache" / arm / f"{op}_kernel.json"
    source_hash = sha256_path(source)
    entries_hash = entry_files_hash(entry_files)
    cached = read_json(cache_path, {})
    if (
        not force
        and isinstance(cached, dict)
        and cached.get("project") == project
        and cached.get("arm") == arm
        and cached.get("source_hash") == source_hash
        and cached.get("entries_hash") == entries_hash
        and cached.get("entry_count") == len(entry_files)
    ):
        return cached["measurement"]

    build_dir = cache_root / "build" / arm / op
    build_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, build_dir / "kernel.cu")
    measurement = CUDABackend().profile_kernel(
        {
            "tmp_dir": build_dir,
            "io_dir": io_dir,
            "entry_files": [path.name for path in entry_files],
        },
        baseline=True,
    )
    write_json(
        cache_path,
        {
            "project": project,
            "arm": arm,
            "op": op,
            "source": str(source),
            "source_hash": source_hash,
            "entries_hash": entries_hash,
            "entry_count": len(entry_files),
            "measurement": measurement,
            "created_at_utc": utc_now(),
        },
    )
    return measurement


HIGH_LEVEL_FALLBACK_PATTERNS = [
    r"\bat::conv1d\b",
    r"\bat::conv2d\b",
    r"\bat::convolution\b",
    r"\bat::_convolution\b",
    r"\bat::batch_norm\b",
    r"\bat::native_batch_norm\b",
    r"\bat::linear\b",
    r"\bat::matmul\b",
    r"\bat::mm\b",
    r"\bat::relu\b",
    r"\bat::gelu\b",
    r"\bat::silu\b",
    r"\bat::sigmoid\b",
    r"\bat::softmax\b",
    r"\bat::softplus\b",
    r"\bat::pad\b",
    r"\bat::embedding\b",
    r"\bat::max_pool2d\b",
    r"\bat::adaptive_avg_pool2d\b",
    r"\bat::layer_norm\b",
    r"\bat::group_norm\b",
    r"\bat::scaled_dot_product_attention\b",
    r"\btorch::nn::functional\b",
]


def real_cuda_classification(source: str | None) -> str:
    if not source:
        return "missing"
    path = Path(source)
    if not path.exists():
        return "unknown"
    text = path.read_text(encoding="utf-8", errors="ignore")
    has_global = "__global__" in text
    has_launch = "<<<" in text
    has_fallback = any(re.search(pattern, text) for pattern in HIGH_LEVEL_FALLBACK_PATTERNS)
    if has_global and has_launch and not has_fallback:
        return "yes"
    if has_global and has_launch and has_fallback:
        return "mixed"
    if has_fallback:
        return "no"
    return "no"


def replay_arm(
    *,
    project: str,
    arm: str,
    device: str,
    force: bool,
    prefer_manifest: bool,
) -> dict[str, Any]:
    from src.optimizer.benchmarking.benchmark_ops import _load_unique_case_profiles

    project_dir = REPO_ROOT / "kernels" / "projects" / project
    cache_root = RUNS_ROOT / project
    selected = selected_kernel_map(project_dir, arm, prefer_manifest=prefer_manifest)
    summary_path = project_dir / "io" / "summary.json"
    case_profiles = _load_unique_case_profiles(summary_path)
    by_op_rows: list[dict[str, Any]] = []
    by_case_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for op in discovered_ops(project_dir):
        io_dir = project_dir / "io" / "individual_ops" / op
        entry_files = sorted(io_dir.glob("entry_*.pt"))
        if not entry_files:
            continue

        case_profile = case_profiles.get(op, {})
        counts_by_entry = case_profile.get("case_counts_by_entry")
        if not isinstance(counts_by_entry, dict) or not counts_by_entry:
            counts_by_entry = {path.name: 1 for path in entry_files}
        counts_by_entry = {str(key): int(value) for key, value in counts_by_entry.items()}

        try:
            torch_measurement = measure_torch_op(
                cache_root=cache_root,
                project=project,
                op=op,
                entry_files=entry_files,
                device=device,
                force=force,
            )
            torch_weighted = weighted_ms(torch_measurement, counts_by_entry)
        except Exception as exc:
            errors.append(f"{arm}/{op}: torch replay failed: {exc}")
            continue

        kernel_path_text = selected.get(op)
        kernel_measurement = None
        kernel_weighted: dict[str, Any] | None = None
        kernel_status = "missing_selected_kernel"
        if kernel_path_text:
            source = Path(kernel_path_text)
            try:
                kernel_measurement = profile_kernel_source(
                    cache_root=cache_root,
                    project=project,
                    arm=arm,
                    op=op,
                    source=source,
                    io_dir=io_dir,
                    entry_files=entry_files,
                    force=force,
                )
                kernel_weighted = weighted_ms(kernel_measurement, counts_by_entry)
                kernel_status = "ok" if kernel_weighted["complete"] else "partial"
            except Exception as exc:
                kernel_status = "replay_failed"
                errors.append(f"{arm}/{op}: kernel replay failed: {exc}")

        torch_total_ms = (
            float(torch_weighted["avg_ms"]) * float(torch_weighted["expected_weight"])
            if torch_weighted["avg_ms"] is not None
            else None
        )
        kernel_total_ms = None
        speedup = None
        if kernel_weighted and kernel_weighted["avg_ms"] is not None:
            kernel_total_ms = float(kernel_weighted["avg_ms"]) * float(kernel_weighted["expected_weight"])
            if kernel_total_ms > 0 and torch_total_ms is not None:
                speedup = torch_total_ms / kernel_total_ms

        profile_kernel_median_ms = None
        if isinstance(kernel_measurement, dict):
            raw = kernel_measurement.get("median_time_ms") or kernel_measurement.get("mean_time_ms")
            profile_kernel_median_ms = float(raw) if raw is not None else None

        by_op_rows.append(
            {
                "project": project,
                "arm": arm,
                "op": op,
                "selected_kernel": kernel_path_text or "",
                "selected_kernel_file": Path(kernel_path_text).name if kernel_path_text else "",
                "real_cuda": real_cuda_classification(kernel_path_text),
                "kernel_status": kernel_status,
                "unique_cases": int(case_profile.get("unique_cases", len(entry_files))),
                "trace_calls": int(case_profile.get("total_calls", sum(counts_by_entry.values()))),
                "eager_weighted_ms": torch_total_ms,
                "forge_weighted_ms": kernel_total_ms,
                "speedup_vs_eager": speedup,
                "mixed_choice": "forge" if speedup and speedup > 1.0 else "torch",
                "torch_missing_entries": ";".join(torch_weighted["missing_entries"]),
                "kernel_missing_entries": ";".join(kernel_weighted["missing_entries"]) if kernel_weighted else "",
                "weighted_eager_avg_ms_per_call": torch_weighted["avg_ms"],
                "weighted_forge_avg_ms_per_call": kernel_weighted["avg_ms"] if kernel_weighted else None,
                "profile_kernel_median_ms": profile_kernel_median_ms,
            }
        )

        torch_latencies = torch_weighted["latency_by_entry"]
        kernel_latencies = kernel_weighted["latency_by_entry"] if kernel_weighted else {}
        for entry_file, count in sorted(counts_by_entry.items()):
            torch_ms = torch_latencies.get(entry_file)
            forge_ms = kernel_latencies.get(entry_file)
            by_case_rows.append(
                {
                    "project": project,
                    "arm": arm,
                    "op": op,
                    "entry_file": entry_file,
                    "trace_calls": count,
                    "eager_ms": torch_ms,
                    "forge_ms": forge_ms,
                    "speedup_vs_eager": (torch_ms / forge_ms) if torch_ms and forge_ms and forge_ms > 0 else None,
                }
            )

    eager_total = sum(float(row["eager_weighted_ms"] or 0.0) for row in by_op_rows)
    forge_rows = [
        row for row in by_op_rows if row.get("forge_weighted_ms") is not None and row.get("kernel_status") == "ok"
    ]
    forge_selected_eager_total = sum(float(row["eager_weighted_ms"] or 0.0) for row in forge_rows)
    forge_selected_total = sum(float(row["forge_weighted_ms"] or 0.0) for row in forge_rows)
    mixed_total = 0.0
    for row in by_op_rows:
        eager_ms = float(row["eager_weighted_ms"] or 0.0)
        forge_ms_raw = row.get("forge_weighted_ms")
        if forge_ms_raw is None or row.get("kernel_status") != "ok":
            mixed_total += eager_ms
            continue
        mixed_total += min(eager_ms, float(forge_ms_raw))

    for row in by_op_rows:
        row["runtime_responsibility"] = (
            float(row["eager_weighted_ms"] or 0.0) / eager_total if eager_total > 0 else None
        )

    out_dir = cache_root / arm
    write_csv(out_dir / f"{arm}_by_op.csv", by_op_rows)
    write_csv(out_dir / f"{arm}_by_unique_case.csv", by_case_rows)
    result = {
        "project": project,
        "arm": arm,
        "formula": "sum(trace_count * eager_ms_per_unique_case) / sum(trace_count * forge_ms_per_unique_case)",
        "selected_kernel_map_for_arm": selected,
        "eager_weighted_ms": eager_total,
        "forge_selected_eager_weighted_ms": forge_selected_eager_total,
        "forge_selected_weighted_ms": forge_selected_total,
        "forge_selected_speedup": (
            forge_selected_eager_total / forge_selected_total if forge_selected_total > 0 else None
        ),
        "mixed_best_of_torch_and_forge_weighted_ms": mixed_total,
        "mixed_best_of_torch_and_forge_speedup": eager_total / mixed_total if mixed_total > 0 else None,
        "selected_kernel_count": len(selected),
        "profiled_op_count": len(discovered_ops(project_dir)),
        "op_rows": by_op_rows,
        "case_rows": by_case_rows,
        "errors": errors,
        "created_at_utc": utc_now(),
    }
    write_json(out_dir / f"{arm}_trace_weighted.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace-weighted replay for unique kernel projects.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-manifest-selection", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = []
    for arm in args.arms:
        print(f"[{utc_now()}] replay start project={args.project} arm={arm}", flush=True)
        result = replay_arm(
            project=args.project,
            arm=arm,
            device=args.device,
            force=args.force,
            prefer_manifest=not args.ignore_manifest_selection,
        )
        print(
            f"[{utc_now()}] replay done project={args.project} arm={arm} "
            f"speedup={result['forge_selected_speedup']} errors={len(result['errors'])}",
            flush=True,
        )
        summaries.append(
            {
                "project": result["project"],
                "arm": result["arm"],
                "eager_weighted_ms": result["eager_weighted_ms"],
                "forge_selected_weighted_ms": result["forge_selected_weighted_ms"],
                "forge_selected_speedup": result["forge_selected_speedup"],
                "mixed_best_of_torch_and_forge_weighted_ms": result[
                    "mixed_best_of_torch_and_forge_weighted_ms"
                ],
                "mixed_best_of_torch_and_forge_speedup": result[
                    "mixed_best_of_torch_and_forge_speedup"
                ],
                "selected_kernel_count": result["selected_kernel_count"],
                "profiled_op_count": result["profiled_op_count"],
                "error_count": len(result["errors"]),
            }
        )

    project_root = RUNS_ROOT / args.project
    write_csv(project_root / "trace_weighted_summary.csv", summaries)
    write_json(
        project_root / "trace_weighted_summary.json",
        {"project": args.project, "created_at_utc": utc_now(), "arms": summaries},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
