from __future__ import annotations

import ast
import hashlib
import json
import os
import sqlite3
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PYTORCH_SELECTION = "__PYTORCH__"
SELECTION_POLICY = "auto_best_fastest_valid"
SUPPORTED_SOURCE_SUFFIXES = {".cu"}


def _repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_path(value: str | os.PathLike[str] | Path) -> Path:
    return Path(value).expanduser().resolve()


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed > 0.0:
        return parsed
    return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _repo_relpath(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_selected_path(raw: Any, project_dir: Path, repo_root: Path) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text == PYTORCH_SELECTION:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate.resolve()
    for base in (repo_root, project_dir):
        resolved = (base / text).resolve()
        if resolved.exists():
            return resolved
    return (repo_root / text).resolve()


def _kernel_source_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES


def _load_benchmark_rows(project_dir: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(project_dir / "benchmarks" / "op_benchmarks.json")
    rows = payload.get("results") if isinstance(payload, dict) else None
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        op_name = str(row.get("op") or "").strip()
        if op_name:
            out[op_name] = row
    return out


def _project_ops(
    project_dir: Path,
    benchmark_rows: dict[str, dict[str, Any]],
    selected_kernels: dict[str, Any],
) -> list[str]:
    ops = set(benchmark_rows.keys())
    ops.update(str(op) for op in selected_kernels.keys())
    for base in (
        project_dir / "io" / "individual_ops",
        project_dir / "trees",
        project_dir / "kernels" / "generated" / "individual_op_kernels",
    ):
        if not base.exists():
            continue
        for child in base.iterdir():
            if child.is_dir():
                ops.add(child.name)
    return sorted(op for op in ops if op)


def _candidate_payload(
    *,
    op_name: str,
    source_path: Path,
    repo_root: Path,
    runtime_ms: float | None,
    pytorch_ms: float | None,
    evidence_tier: str,
    source_origin: str,
    selection_reason: str,
    paper_eligible: bool = False,
    node_id: int | None = None,
) -> dict[str, Any]:
    relpath = _repo_relpath(source_path, repo_root)
    payload: dict[str, Any] = {
        "op": op_name,
        "candidate_id": f"{op_name}:{relpath}",
        "kernel_source_path": str(source_path.resolve()),
        "kernel_source_repo_relpath": relpath,
        "kernel_source_sha256": _sha256_file(source_path),
        "median_latency_ms": runtime_ms,
        "runtime_ms": runtime_ms,
        "pytorch_ms": pytorch_ms,
        "evidence_tier": evidence_tier,
        "source_origin": source_origin,
        "selection_reason": selection_reason,
        "paper_eligible": bool(paper_eligible),
    }
    if node_id is not None:
        payload["node_id"] = int(node_id)
    if runtime_ms is not None and pytorch_ms is not None and runtime_ms > 0.0:
        payload["speedup"] = float(pytorch_ms / runtime_ms)
    return payload


def _resolve_tree_kernel_path(
    project_dir: Path,
    repo_root: Path,
    op_name: str,
    node_id: int,
    code_ref: Any,
) -> Path | None:
    candidates: list[Path] = []
    code_text = str(code_ref or "").strip()
    if code_text:
        code_path = Path(code_text)
        if code_path.is_absolute():
            candidates.append(code_path)
        else:
            candidates.extend(
                [
                    project_dir / "trees" / code_text,
                    project_dir / code_text,
                    repo_root / code_text,
                ]
            )

    kernel_dir = project_dir / "trees" / op_name / "kernels"
    candidates.extend(
        [
            kernel_dir / f"kernel_{node_id}.cu",
            kernel_dir / f"kernel_{node_id}.py",
            kernel_dir / f"kernel_{node_id}.metal",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _tree_candidates(
    project_dir: Path,
    repo_root: Path,
    op_name: str,
    pytorch_ms: float | None,
) -> list[dict[str, Any]]:
    db_path = project_dir / "trees" / op_name / "nodes.db"
    if not db_path.exists():
        return []

    candidates: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(nodes)").fetchall()}
            metric_col = "median_time_ms" if "median_time_ms" in columns else "value"
            if "code" in columns:
                rows = conn.execute(
                    f"SELECT id, {metric_col}, code FROM nodes WHERE {metric_col} IS NOT NULL AND {metric_col} > 0"
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT id, {metric_col}, NULL FROM nodes WHERE {metric_col} IS NOT NULL AND {metric_col} > 0"
                ).fetchall()
    except Exception:
        return []

    for node_id_raw, runtime_raw, code_ref in rows:
        runtime_ms = _positive_float(runtime_raw)
        if runtime_ms is None:
            continue
        try:
            node_id = int(node_id_raw)
        except Exception:
            continue
        source_path = _resolve_tree_kernel_path(project_dir, repo_root, op_name, node_id, code_ref)
        if source_path is None or not _kernel_source_supported(source_path):
            continue
        candidates.append(
            _candidate_payload(
                op_name=op_name,
                source_path=source_path,
                repo_root=repo_root,
                runtime_ms=runtime_ms,
                pytorch_ms=pytorch_ms,
                evidence_tier="operator",
                source_origin="optimization_tree",
                selection_reason="fastest valid optimization-tree kernel",
                paper_eligible=False,
                node_id=node_id,
            )
        )
    return candidates


def _generated_candidate(
    project_dir: Path,
    repo_root: Path,
    op_name: str,
    runtime_ms: float | None,
    pytorch_ms: float | None,
    source_origin: str,
) -> dict[str, Any] | None:
    source_path = (
        project_dir
        / "kernels"
        / "generated"
        / "individual_op_kernels"
        / op_name
        / "kernel.cu"
    ).resolve()
    if not source_path.exists():
        return None
    return _candidate_payload(
        op_name=op_name,
        source_path=source_path,
        repo_root=repo_root,
        runtime_ms=runtime_ms,
        pytorch_ms=pytorch_ms,
        evidence_tier="operator",
        source_origin=source_origin,
        selection_reason="benchmarked generated kernel",
        paper_eligible=False,
    )


def _benchmark_candidate(
    project_dir: Path,
    repo_root: Path,
    op_name: str,
    row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    runtime_ms = _positive_float(row.get("weighted_kernel_ms")) or _positive_float(row.get("kernel_ms"))
    pytorch_ms = _positive_float(row.get("weighted_pytorch_ms")) or _positive_float(row.get("pytorch_ms"))
    source_raw = row.get("kernel_source_path")
    source_path = _resolve_selected_path(source_raw, project_dir, repo_root) if source_raw else None
    if source_path is None or not source_path.exists():
        source_origin = str(row.get("kernel_source_origin") or "generated_root")
        return _generated_candidate(project_dir, repo_root, op_name, runtime_ms, pytorch_ms, source_origin)
    if not _kernel_source_supported(source_path):
        return None
    return _candidate_payload(
        op_name=op_name,
        source_path=source_path,
        repo_root=repo_root,
        runtime_ms=runtime_ms,
        pytorch_ms=pytorch_ms,
        evidence_tier="operator",
        source_origin=str(row.get("kernel_source_origin") or "benchmark_results"),
        selection_reason="fastest benchmarked kernel from op_benchmarks.json",
        paper_eligible=False,
    )


def _candidate_rejection_reasons(
    candidate: dict[str, Any],
    row: dict[str, Any] | None,
    *,
    allow_operator_only: bool,
    unsafe_override: bool,
) -> list[str]:
    if unsafe_override:
        return []
    reasons: list[str] = []
    source_path = Path(str(candidate.get("kernel_source_path") or ""))
    if not source_path.exists():
        reasons.append("missing_kernel_source")
    elif not _kernel_source_supported(source_path):
        reasons.append("unsupported_kernel_source")
    runtime_ms = _positive_float(candidate.get("median_latency_ms"))
    if runtime_ms is None:
        reasons.append("missing_runtime_ms")
    if isinstance(row, dict):
        status = str(row.get("kernel_status") or "")
        if status and status != "ok":
            reasons.append(f"kernel_status:{status}")
        if row.get("kernel_estimated"):
            reasons.append("estimated_runtime")
        pytorch_ms = _positive_float(row.get("weighted_pytorch_ms")) or _positive_float(row.get("pytorch_ms"))
        winner = str(row.get("winner") or "")
        if (
            runtime_ms is not None
            and pytorch_ms is not None
            and runtime_ms >= pytorch_ms
            and winner != "optimized"
        ):
            reasons.append("not_faster_than_pytorch")
    if not allow_operator_only and str(candidate.get("evidence_tier") or "") == "operator":
        reasons.append("operator_only_evidence_disabled")
    return reasons


def _sort_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[float, str, str]:
        runtime_ms = _positive_float(item.get("median_latency_ms"))
        runtime_sort = runtime_ms if runtime_ms is not None else 1e300
        return (
            runtime_sort,
            str(item.get("source_origin") or ""),
            str(item.get("kernel_source_repo_relpath") or ""),
        )

    return sorted(candidates, key=key)


def _candidate_matches_selected(candidate: dict[str, Any], selected_path: Path, repo_root: Path) -> bool:
    candidate_path = Path(str(candidate.get("kernel_source_path") or "")).resolve()
    if candidate_path == selected_path.resolve():
        return True
    selected_rel = _repo_relpath(selected_path, repo_root)
    return str(candidate.get("kernel_source_repo_relpath") or "") == selected_rel


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def resolve_cast_export_plan(
    project_dir: str | os.PathLike[str] | Path,
    *,
    repo_root: str | os.PathLike[str] | Path | None = None,
    selected_kernels: dict[str, Any] | None = None,
    allow_operator_only: bool = True,
    unsafe_override: bool = False,
    allow_native_package: bool = False,
) -> dict[str, Any]:
    repo = _as_path(repo_root) if repo_root is not None else _repo_root_from_module()
    project = _as_path(project_dir)
    selected = selected_kernels if isinstance(selected_kernels, dict) else {}
    benchmark_rows = _load_benchmark_rows(project)

    selected_ops: dict[str, dict[str, Any]] = {}
    selected_kernel_map: dict[str, str] = {}
    rejected_candidates: dict[str, list[dict[str, Any]]] = {}
    skipped_ops: dict[str, str] = {}

    for op_name in _project_ops(project, benchmark_rows, selected):
        selected_raw = selected.get(op_name)
        if str(selected_raw or "") == PYTORCH_SELECTION:
            skipped_ops[op_name] = "selected_pytorch_baseline"
            continue

        row = benchmark_rows.get(op_name)
        candidates: list[dict[str, Any]] = []
        bench_candidate = _benchmark_candidate(project, repo, op_name, row)
        if bench_candidate is not None:
            candidates.append(bench_candidate)
        pytorch_ms = (
            _positive_float(row.get("weighted_pytorch_ms")) or _positive_float(row.get("pytorch_ms"))
            if isinstance(row, dict)
            else None
        )
        candidates.extend(_tree_candidates(project, repo, op_name, pytorch_ms))

        deduped: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            path_key = str(Path(str(candidate.get("kernel_source_path") or "")).resolve())
            current = deduped.get(path_key)
            if current is None:
                deduped[path_key] = candidate
                continue
            current_ms = _positive_float(current.get("median_latency_ms"))
            candidate_ms = _positive_float(candidate.get("median_latency_ms"))
            if current_ms is None or (candidate_ms is not None and candidate_ms < current_ms):
                deduped[path_key] = candidate

        valid: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for candidate in deduped.values():
            reasons = _candidate_rejection_reasons(
                candidate,
                row,
                allow_operator_only=allow_operator_only,
                unsafe_override=unsafe_override,
            )
            if reasons:
                rejected.append({**candidate, "rejection_reasons": reasons})
            else:
                valid.append(candidate)

        valid = _sort_candidates(valid)
        selected_candidate = valid[0] if valid else None
        selected_path = _resolve_selected_path(selected_raw, project, repo)
        if selected_path is not None:
            matching = [candidate for candidate in valid if _candidate_matches_selected(candidate, selected_path, repo)]
            if matching and (unsafe_override or matching[0] == selected_candidate):
                selected_candidate = matching[0]
                if matching[0] != valid[0]:
                    selected_candidate = {
                        **selected_candidate,
                        "evidence_tier": "manual_override",
                        "selection_reason": "manual export selection",
                        "paper_eligible": False,
                    }
            elif matching:
                rejected.append(
                    {
                        **matching[0],
                        "rejection_reasons": ["manual_selection_not_fastest_valid"],
                    }
                )

        if selected_candidate is not None:
            selected_ops[op_name] = selected_candidate
            selected_kernel_map[op_name] = str(Path(selected_candidate["kernel_source_path"]).resolve())
        if rejected:
            rejected_candidates[op_name] = rejected
        if selected_candidate is None and not rejected:
            skipped_ops[op_name] = "no_exportable_cuda_kernel"

    rejected_summary: dict[str, int] = {}
    for rows in rejected_candidates.values():
        for item in rows:
            for reason in item.get("rejection_reasons", []):
                rejected_summary[reason] = int(rejected_summary.get(reason, 0)) + 1

    exportable = bool(selected_ops) or bool(allow_native_package)
    return {
        "selection_policy": SELECTION_POLICY,
        "selection_policy_details": {
            "policy_name": SELECTION_POLICY,
            "allow_operator_only": bool(allow_operator_only),
            "unsafe_override": bool(unsafe_override),
            "allow_native_package": bool(allow_native_package),
        },
        "project_id": project.name,
        "project_root": str(project),
        "project_ref": project.name,
        "git_commit": _git_commit(repo),
        "selected_ops": selected_ops,
        "selected_op_count": len(selected_ops),
        "selected_kernel_map": selected_kernel_map,
        "exportable": exportable,
        "export_paper_eligible": bool(selected_ops) and all(
            bool(item.get("paper_eligible")) for item in selected_ops.values()
        ),
        "rejected_candidates": rejected_candidates,
        "rejected_candidate_summary": rejected_summary,
        "skipped_ops": skipped_ops,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_cast_manifest_metadata(selection_plan: dict[str, Any]) -> dict[str, Any]:
    selected_ops = selection_plan.get("selected_ops") if isinstance(selection_plan, dict) else {}
    if not isinstance(selected_ops, dict):
        selected_ops = {}
    return {
        "selection_policy_details": selection_plan.get("selection_policy_details", {}),
        "selected_ops": sorted(selected_ops.keys()),
        "selected_op_count": len(selected_ops),
        "selected_kernel_map": {
            op_name: str(entry.get("kernel_source_repo_relpath") or entry.get("kernel_source_path") or "")
            for op_name, entry in selected_ops.items()
            if isinstance(entry, dict)
        },
        "selected_kernel_metadata": selected_ops,
    }


def _model_class_name(model_py_bytes: bytes) -> str:
    try:
        tree = ast.parse(model_py_bytes.decode("utf-8"))
    except Exception:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return node.name
    return ""


def _weight_file(project_dir: Path) -> tuple[str, bytes, int]:
    config = _read_json(project_dir / "config.json")
    if isinstance(config, dict):
        artifacts = config.get("artifacts")
        weights = artifacts.get("weights") if isinstance(artifacts, dict) else None
        if isinstance(weights, list):
            for artifact in weights:
                relpath = artifact.get("relpath") if isinstance(artifact, dict) else None
                if not relpath:
                    continue
                candidate = (project_dir / str(relpath)).resolve()
                if candidate.exists():
                    data = candidate.read_bytes()
                    sha = _sha256_bytes(data)
                    return f"weights/{sha}.pt", data, len(data)
    for name in ("weights.pt", "model.pt"):
        candidate = project_dir / name
        if candidate.exists():
            data = candidate.read_bytes()
            sha = _sha256_bytes(data)
            return f"weights/{sha}.pt", data, len(data)
    return "", b"", 0


def _sm_version() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return f"sm_{cap[0]}{cap[1]}"
    except Exception:
        pass
    return ""


def _build_archive_file_map(
    project_dir: Path,
    repo_root: Path,
    selection_plan: dict[str, Any],
) -> tuple[dict[str, bytes], list[dict[str, Any]], str, int, str]:
    file_map: dict[str, bytes] = {}
    model_py = project_dir / "model.py"
    if not model_py.exists():
        raise FileNotFoundError(f"Missing model.py at {model_py}")
    model_py_bytes = model_py.read_bytes()
    model_class_name = _model_class_name(model_py_bytes)
    file_map["model.py"] = model_py_bytes

    model_config = project_dir / "model_config.json"
    if model_config.exists():
        file_map["model_config.json"] = model_config.read_bytes()

    weight_archive_path, weight_data, weight_size = _weight_file(project_dir)
    if weight_archive_path:
        file_map[weight_archive_path] = weight_data

    selected_ops = selection_plan.get("selected_ops") if isinstance(selection_plan, dict) else {}
    if not isinstance(selected_ops, dict):
        selected_ops = {}

    sm_version = _sm_version()
    ops_manifest: list[dict[str, Any]] = []
    for op_name in sorted(selected_ops.keys()):
        selection_entry = selected_ops[op_name]
        if not isinstance(selection_entry, dict):
            continue
        source_path = Path(str(selection_entry.get("kernel_source_path") or "")).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Selected kernel for {op_name} is missing: {source_path}")
        if not _kernel_source_supported(source_path):
            raise RuntimeError(f"Selected kernel for {op_name} is not a CUDA source: {source_path}")

        cu_path = f"kernels/{op_name}/kernel.cu"
        wrapper_path = f"kernels/{op_name}/wrapper.py"
        file_map[cu_path] = source_path.read_bytes()
        file_map[wrapper_path] = (
            f"# Cast dispatch wrapper for {op_name}\n# Generated by KernelForge\n"
        ).encode("utf-8")

        precompiled: dict[str, str] = {}
        if sm_version:
            so_file = source_path.parent / f"{op_name}.so"
            if so_file.exists():
                so_archive_path = f"compiled/{sm_version}/{op_name}.so"
                file_map[so_archive_path] = so_file.read_bytes()
                precompiled[sm_version] = so_archive_path

        ops_manifest.append(
            {
                "name": op_name,
                "kernel_dir": f"kernels/{op_name}/",
                "cuda_source": cu_path,
                "wrapper": wrapper_path,
                "precompiled": precompiled,
                "selection_evidence": selection_entry,
            }
        )

    file_map["loader.py"] = b"# Cast vendored runtime loader\n# pip install cast for the full runtime\n"
    return file_map, ops_manifest, weight_archive_path, weight_size, model_class_name


def export_cast_package(
    project_dir: str | os.PathLike[str] | Path,
    *,
    export_plan: dict[str, Any] | None = None,
    selection_policy: str = SELECTION_POLICY,
    selected_kernels: dict[str, Any] | None = None,
    allow_operator_only: bool = True,
    unsafe_override: bool = False,
    allow_native_package: bool = False,
    repo_root: str | os.PathLike[str] | Path | None = None,
) -> dict[str, Any]:
    repo = _as_path(repo_root) if repo_root is not None else _repo_root_from_module()
    project = _as_path(project_dir)
    if selection_policy != SELECTION_POLICY:
        raise ValueError(f"Unsupported cast export selection policy: {selection_policy}")

    selection_plan = export_plan or resolve_cast_export_plan(
        project,
        repo_root=repo,
        selected_kernels=selected_kernels,
        allow_operator_only=allow_operator_only,
        unsafe_override=unsafe_override,
        allow_native_package=allow_native_package,
    )
    if not selection_plan.get("exportable") and not allow_native_package:
        return {
            "success": False,
            "error": "No kernels satisfied auto_best_fastest_valid. Review rejected candidates.",
            "selection_report": selection_plan,
            "rejected_candidates": selection_plan.get("rejected_candidates", {}),
        }

    exported_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    file_map, ops_manifest, weight_archive_path, weight_size, model_class_name = _build_archive_file_map(
        project,
        repo,
        selection_plan,
    )

    metadata = build_cast_manifest_metadata(selection_plan)
    all_precompiled_sms = sorted({sm for op in ops_manifest for sm in op["precompiled"].keys()})
    manifest_obj = {
        "project_name": project.name,
        "project_id": selection_plan.get("project_id", project.name),
        "project_root": selection_plan.get("project_root", str(project)),
        "project_ref": selection_plan.get("project_ref", project.name),
        "exported_at": exported_at,
        "timestamp": exported_at,
        "git_commit": selection_plan.get("git_commit"),
        "model_class": model_class_name,
        "model_init_args": {},
        "weight_file": weight_archive_path,
        "ops": ops_manifest,
        "selection_policy": selection_plan.get("selection_policy", selection_policy),
        "selection_policy_details": metadata.get("selection_policy_details", {}),
        "selected_ops": metadata.get("selected_ops", []),
        "selected_op_count": metadata.get("selected_op_count", len(ops_manifest)),
        "selected_kernel_map": metadata.get("selected_kernel_map", {}),
        "selected_kernel_metadata": metadata.get("selected_kernel_metadata", {}),
        "export_paper_eligible": bool(selection_plan.get("export_paper_eligible")),
        "rejected_candidate_summary": selection_plan.get("rejected_candidate_summary", {}),
        "skipped_ops": selection_plan.get("skipped_ops", {}),
    }
    file_map["manifest.json"] = json.dumps(manifest_obj, indent=2).encode("utf-8")
    file_map["selection_manifest.json"] = json.dumps(selection_plan, indent=2).encode("utf-8")

    checksum_lines = []
    for archive_path in file_map:
        checksum_lines.append(f"{_sha256_bytes(file_map[archive_path])}  {archive_path}")
    checksums_bytes = "\n".join(checksum_lines).encode("utf-8")
    archive_checksum = _sha256_bytes(checksums_bytes)
    file_map["checksums.sha256"] = checksums_bytes

    header_obj = {
        "format_version": "1.0",
        "file_type": "kernelforge_inference",
        "project_name": project.name,
        "project_ref": selection_plan.get("project_ref", project.name),
        "exported_at": exported_at,
        "kernelforge_version": "0.1.0",
        "git_commit": selection_plan.get("git_commit"),
        "runtime": {
            "min_cast_version": "0.1",
            "min_torch_version": "2.1.0",
            "min_cuda_version": "12.0",
            "target_sm_versions": [],
        },
        "contents": {
            "optimized_op_count": len(ops_manifest),
            "total_op_count": len(ops_manifest),
            "has_precompiled": bool(all_precompiled_sms),
            "precompiled_sm_versions": all_precompiled_sms,
            "weight_size_bytes": weight_size,
        },
        "archive_checksum": archive_checksum,
    }
    header_bytes = json.dumps(header_obj, indent=2).encode("utf-8")

    export_dir = project / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    cast_path = export_dir / f"{project.name}.cast"
    with zipfile.ZipFile(cast_path, "w") as zipf:
        zipf.writestr("HEADER.json", header_bytes)
        for archive_path, data in file_map.items():
            zipf.writestr(archive_path, data)

    return {
        "success": True,
        "name": f"{project.name}.cast",
        "size": cast_path.stat().st_size,
        "export_path": str(cast_path),
        "selection_report": selection_plan,
    }
