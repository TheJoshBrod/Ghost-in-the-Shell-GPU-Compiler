from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

from kernelforge.run_cast import verify_checksums
from paper_benchmarks.paper_bench.cast_export import (
    PYTORCH_SELECTION,
    export_cast_package,
    resolve_cast_export_plan,
)


def _write_project(repo_root: Path, project_name: str = "demo") -> Path:
    project_dir = repo_root / "kernels" / "projects" / project_name
    project_dir.mkdir(parents=True)
    (project_dir / "model.py").write_text(
        "import torch\n"
        "class DemoModel(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x\n",
        encoding="utf-8",
    )
    return project_dir


def _write_kernel(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#include <torch/extension.h>\n"
        "torch::Tensor launch(torch::Tensor x) { return x; }\n",
        encoding="utf-8",
    )
    return path


def _write_benchmarks(project_dir: Path, rows: list[dict]) -> None:
    bench_dir = project_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "op_benchmarks.json").write_text(
        json.dumps({"results": rows}, indent=2),
        encoding="utf-8",
    )


def test_resolve_cast_export_plan_selects_fast_valid_kernel(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    fast_kernel = _write_kernel(project_dir / "trees" / "torch_nn_functional_linear" / "kernels" / "kernel_2.cu")
    slow_kernel = _write_kernel(project_dir / "trees" / "torch_nn_functional_relu" / "kernels" / "kernel_1.cu")
    _write_benchmarks(
        project_dir,
        [
            {
                "op": "torch_nn_functional_linear",
                "kernel_status": "ok",
                "winner": "optimized",
                "kernel_ms": 1.5,
                "pytorch_ms": 4.5,
                "kernel_source_path": str(fast_kernel),
                "kernel_source_origin": "optimized_tree",
            },
            {
                "op": "torch_nn_functional_relu",
                "kernel_status": "ok",
                "winner": "pytorch",
                "kernel_ms": 6.0,
                "pytorch_ms": 3.0,
                "kernel_source_path": str(slow_kernel),
                "kernel_source_origin": "optimized_tree",
            },
        ],
    )

    plan = resolve_cast_export_plan(project_dir, repo_root=tmp_path)

    assert plan["exportable"] is True
    assert sorted(plan["selected_ops"]) == ["torch_nn_functional_linear"]
    selected = plan["selected_ops"]["torch_nn_functional_linear"]
    assert selected["median_latency_ms"] == 1.5
    assert selected["kernel_source_repo_relpath"].endswith("kernel_2.cu")
    assert plan["rejected_candidate_summary"]["not_faster_than_pytorch"] == 1


def test_resolve_cast_export_plan_honors_pytorch_skip(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    kernel = _write_kernel(project_dir / "trees" / "torch_nn_functional_linear" / "kernels" / "kernel_0.cu")
    _write_benchmarks(
        project_dir,
        [
            {
                "op": "torch_nn_functional_linear",
                "kernel_status": "ok",
                "winner": "optimized",
                "kernel_ms": 1.0,
                "pytorch_ms": 5.0,
                "kernel_source_path": str(kernel),
            }
        ],
    )

    plan = resolve_cast_export_plan(
        project_dir,
        repo_root=tmp_path,
        selected_kernels={"torch_nn_functional_linear": PYTORCH_SELECTION},
    )

    assert plan["selected_ops"] == {}
    assert plan["skipped_ops"]["torch_nn_functional_linear"] == "selected_pytorch_baseline"


def test_resolve_cast_export_plan_uses_fastest_tree_fallback(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    op_name = "torch_nn_functional_linear"
    _write_kernel(project_dir / "trees" / op_name / "kernels" / "kernel_1.cu")
    _write_kernel(project_dir / "trees" / op_name / "kernels" / "kernel_2.cu")

    db_path = project_dir / "trees" / op_name / "nodes.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, value REAL, code TEXT)")
        conn.execute(
            "INSERT INTO nodes (id, value, code) VALUES (?, ?, ?)",
            (1, 4.0, f"{op_name}/kernels/kernel_1.cu"),
        )
        conn.execute(
            "INSERT INTO nodes (id, value, code) VALUES (?, ?, ?)",
            (2, 2.0, f"{op_name}/kernels/kernel_2.cu"),
        )

    plan = resolve_cast_export_plan(project_dir, repo_root=tmp_path)

    selected = plan["selected_ops"][op_name]
    assert selected["node_id"] == 2
    assert selected["median_latency_ms"] == 2.0
    assert selected["kernel_source_repo_relpath"].endswith("kernel_2.cu")


def test_export_cast_package_writes_loadable_archive_shape(tmp_path: Path) -> None:
    project_dir = _write_project(tmp_path)
    kernel = _write_kernel(project_dir / "trees" / "torch_nn_functional_linear" / "kernels" / "kernel_0.cu")
    _write_benchmarks(
        project_dir,
        [
            {
                "op": "torch_nn_functional_linear",
                "kernel_status": "ok",
                "winner": "optimized",
                "kernel_ms": 1.0,
                "pytorch_ms": 2.0,
                "kernel_source_path": str(kernel),
            }
        ],
    )

    result = export_cast_package(project_dir, repo_root=tmp_path)

    assert result["success"] is True
    cast_path = Path(result["export_path"])
    with zipfile.ZipFile(cast_path) as zf:
        assert zf.namelist()[0] == "HEADER.json"
        verify_checksums(zf)
        manifest = json.loads(zf.read("manifest.json"))
        selection_manifest = json.loads(zf.read("selection_manifest.json"))

    assert manifest["project_name"] == "demo"
    assert manifest["model_class"] == "DemoModel"
    assert manifest["ops"][0]["name"] == "torch_nn_functional_linear"
    assert manifest["ops"][0]["cuda_source"] == "kernels/torch_nn_functional_linear/kernel.cu"
    assert manifest["selected_ops"] == ["torch_nn_functional_linear"]
    assert selection_manifest["selected_op_count"] == 1
