from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT = "sd35-medium-bf16-gb10"
MODEL_SLUG = "sd35-medium-bf16-gb10"
TARGET = 10
OPS = [
    "torch_nn_functional_conv2d",
    "torch_nn_functional_gelu",
    "torch_nn_functional_group_norm",
    "torch_nn_functional_layer_norm",
    "torch_nn_functional_linear",
    "torch_nn_functional_scaled_dot_product_attention",
    "torch_nn_functional_silu",
]
RUNS_ROOT = REPO_ROOT / "paper_benchmarks" / "runs"
ARTIFACT_ROOT = REPO_ROOT / "paper_benchmarks" / "data_collection" / "artifacts" / MODEL_SLUG
PROJECT_ROOT = REPO_ROOT / "kernels" / "projects" / PROJECT
STATUS_DIR = REPO_ROOT / "paper_benchmarks" / "runs" / "sd35_publishable_full44_warm20_driver"
STATUS_PATH = STATUS_DIR / "status.json"
LOG_PATH = STATUS_DIR / "driver.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def log(message: str) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def update_status(**updates: Any) -> None:
    payload = read_json(STATUS_PATH, {})
    if not isinstance(payload, dict):
        payload = {}
    payload.update(updates)
    payload["updated_at_utc"] = utc_now()
    write_json(STATUS_PATH, payload)


def command_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("KFORGE_ALLOW_HF_DOWNLOAD", "0")
    env.setdefault("KFORGE_PROFILE_MAX_PER_OP", "all")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_logged(name: str, cmd: list[str], timeout_s: int | None = None) -> None:
    update_status(active_step=name, active_command=cmd, active_started_at_utc=utc_now())
    log(f"START {name}: {' '.join(cmd)}")
    started = time.perf_counter()
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {name} START {utc_now()} =====\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=command_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        write_json(STATUS_DIR / f"{name}.pid.json", {"pid": proc.pid, "command": cmd, "started_at_utc": utc_now()})
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            rc = 124
        handle.write(f"===== {name} END {utc_now()} rc={rc} elapsed_s={time.perf_counter() - started:.3f} =====\n")
    if rc != 0:
        update_status(active_step=None, failed_step=name, failed_returncode=rc)
        raise RuntimeError(f"{name} failed with exit code {rc}")
    log(f"DONE {name}: elapsed_s={time.perf_counter() - started:.3f}")
    update_status(active_step=None)


def max_node_id(op: str) -> int:
    db_path = PROJECT_ROOT / "trees" / op / "nodes.db"
    if not db_path.exists():
        return -1
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT MAX(id) FROM nodes").fetchone()
    return int(row[0]) if row and row[0] is not None else -1


def tree_state() -> dict[str, int]:
    return {op: max_node_id(op) for op in OPS}


def ensure_opt10() -> None:
    while True:
        state = tree_state()
        update_status(tree_state=state)
        remaining = [op for op, node_id in state.items() if node_id < TARGET]
        if not remaining:
            log(f"opt10 tree target reached: {state}")
            return
        min_existing = min(state[op] for op in remaining)
        iterations = max(1, TARGET - min_existing)
        run_logged(
            "optimize_to_10",
            [
                sys.executable,
                "-m",
                "src.optimizer.workflow",
                "optimize",
                "--project",
                PROJECT,
                "--ops",
                ",".join(remaining),
                "--iterations",
                str(iterations),
                "--workers",
                "1",
                "--llm-model",
                "claude-opus-4-7",
                "--llm-provider",
                "anthropic",
            ],
            timeout_s=7200,
        )


def arm_manifest_exists(arm: str) -> bool:
    return any(
        path.name == "collection_manifest.json" and f"__{arm}__" in str(path.parent)
        for path in ARTIFACT_ROOT.glob(f"*__{arm}__*/collection_manifest.json")
    )


def latest_arm_manifest(arm: str) -> Path:
    manifests = sorted(
        ARTIFACT_ROOT.glob(f"*__{arm}__*/collection_manifest.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not manifests:
        raise FileNotFoundError(f"No collection manifest found for arm {arm}")
    return manifests[-1]


def collect_opt10() -> None:
    if arm_manifest_exists("optimize_10"):
        log(f"optimize_10 artifact already exists: {latest_arm_manifest('optimize_10')}")
        return
    run_logged(
        "collect_optimize_10",
        [
            sys.executable,
            "-m",
            "paper_benchmarks.data_collection.collect_zero_shot",
            "--project",
            PROJECT,
            "--model-slug",
            MODEL_SLUG,
            "--arm",
            "optimize_10",
        ],
        timeout_s=1800,
    )


def cast_path(arm: str, policy: str) -> Path:
    manifest_path = latest_arm_manifest(arm)
    manifest = read_json(manifest_path, {})
    key = f"{policy}_forge_cast"
    artifact = manifest.get(key) if isinstance(manifest, dict) else None
    path_raw = artifact.get("path") if isinstance(artifact, dict) else None
    if not path_raw:
        raise KeyError(f"{manifest_path} does not contain {key}.path")
    path = Path(path_raw)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def summary_exists(run_id: str) -> bool:
    return (RUNS_ROOT / run_id / "summary.json").exists()


def run_external(run_id: str, variants: list[str], cast: Path | None = None, timeout_s: int = 14400) -> None:
    if summary_exists(run_id):
        log(f"SKIP external {run_id}: summary exists")
        return
    cmd = [
        sys.executable,
        "-m",
        "paper_benchmarks.data_collection.run_sd35_external_benchmark",
        "--variants",
        *variants,
        "--max-prompts",
        "44",
        "--warmup-count",
        "20",
        "--run-id",
        run_id,
    ]
    if cast is not None:
        cmd.extend(["--cast-package", str(cast)])
    run_logged(f"external_{run_id}", cmd, timeout_s=timeout_s)


def summarize() -> None:
    cmd = [
        sys.executable,
        "-m",
        "paper_benchmarks.scripts.summarize_sd35_publishable_full44_warm20",
    ]
    script = REPO_ROOT / "paper_benchmarks" / "scripts" / "summarize_sd35_publishable_full44_warm20.py"
    if script.exists():
        run_logged("summarize_results", cmd, timeout_s=300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detached/resumable SD3.5 publishable full44 warm20 benchmark driver.")
    parser.add_argument("--skip-full-forge", action="store_true", help="Skip zero/full and opt5/full long-running rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    update_status(
        created_at_utc=read_json(STATUS_PATH, {}).get("created_at_utc", utc_now()),
        driver_pid=os.getpid(),
        protocol={
            "workload": "paper_benchmarks/workloads/sd35/t2i_compbench_external_v1.jsonl",
            "max_prompts": 44,
            "cold_prompts": 1,
            "warmup_prompts": 20,
            "timed_warm_prompts": 23,
            "model": "stabilityai/stable-diffusion-3.5-medium",
        },
    )
    log("driver started")

    ensure_opt10()
    collect_opt10()

    run_external("sd35_t2i_compbench_eager_full44_warm20", ["eager"], timeout_s=3600)
    run_external("sd35_t2i_compbench_torch_compile_full44_warm20", ["torch_compile"], timeout_s=5400)
    run_external("sd35_t2i_compbench_zero_mixed_full44_warm20", ["kf_cast"], cast_path("zero_shot", "mixed"), timeout_s=3600)
    run_external("sd35_t2i_compbench_opt5_mixed_full44_warm20", ["kf_cast"], cast_path("optimize_5", "mixed"), timeout_s=3600)
    run_external("sd35_t2i_compbench_opt10_mixed_full44_warm20", ["kf_cast"], cast_path("optimize_10", "mixed"), timeout_s=3600)

    if not args.skip_full_forge:
        run_external("sd35_t2i_compbench_zero_full_full44_warm20", ["kf_cast"], cast_path("zero_shot", "full"), timeout_s=21600)
        run_external("sd35_t2i_compbench_opt5_full_full44_warm20", ["kf_cast"], cast_path("optimize_5", "full"), timeout_s=14400)

    summarize()
    update_status(done=True, finished_at_utc=utc_now(), active_step=None)
    log("driver completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
