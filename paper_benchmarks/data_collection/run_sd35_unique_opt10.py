from __future__ import annotations

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
PROJECT = "sd35-medium-bf16-gb10-unique"
MODEL_SLUG = PROJECT
TARGET = 10

RUNS_ROOT = REPO_ROOT / "paper_benchmarks" / "runs"
ARTIFACT_ROOT = REPO_ROOT / "paper_benchmarks" / "data_collection" / "artifacts" / MODEL_SLUG
PROJECT_ROOT = REPO_ROOT / "kernels" / "projects" / PROJECT
STATUS_DIR = RUNS_ROOT / "sd35_unique_opt10_driver"
STATUS_PATH = STATUS_DIR / "status.json"
LOG_PATH = STATUS_DIR / "driver.log"


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
    env.setdefault("KFORGE_PROFILE_CAPTURE_MODE", "unique")
    env.setdefault("KFORGE_TARGET_DEVICE", "cuda")
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
        write_json(
            STATUS_DIR / f"{name}.pid.json",
            {"pid": proc.pid, "command": cmd, "started_at_utc": utc_now()},
        )
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            rc = 124
        handle.write(
            f"===== {name} END {utc_now()} rc={rc} "
            f"elapsed_s={time.perf_counter() - started:.3f} =====\n"
        )
    if rc != 0:
        update_status(active_step=None, failed_step=name, failed_returncode=rc)
        raise RuntimeError(f"{name} failed with exit code {rc}")
    log(f"DONE {name}: elapsed_s={time.perf_counter() - started:.3f}")
    update_status(active_step=None)


def profile_ready() -> bool:
    summary = read_json(PROJECT_ROOT / "io" / "summary.json", {})
    unique_cases = read_json(PROJECT_ROOT / "io" / "unique_cases.json", {})
    if not isinstance(summary, dict) or not isinstance(unique_cases, dict):
        return False
    return (
        summary.get("profile_capture_mode") == "unique"
        and bool(summary.get("op_unique_counts"))
        and bool(unique_cases.get("ops"))
    )


def discovered_ops() -> list[str]:
    io_root = PROJECT_ROOT / "io" / "individual_ops"
    if not io_root.exists():
        return []
    return sorted(path.name for path in io_root.iterdir() if path.is_dir())


def generated_success_ops() -> set[str]:
    root = PROJECT_ROOT / "kernels" / "generated" / "individual_op_kernels"
    if not root.exists():
        return set()
    out: set[str] = set()
    for op_dir in root.iterdir():
        if not op_dir.is_dir():
            continue
        if any((op_dir / marker).exists() for marker in ("success.cuda", "success.triton")):
            out.add(op_dir.name)
    return out


def profile_unique() -> None:
    if profile_ready():
        log("SKIP profile_unique: unique profile already exists")
        return
    run_logged(
        "profile_unique",
        [
            sys.executable,
            "-m",
            "src.optimizer.workflow",
            "profile",
            "--project",
            PROJECT,
        ],
        timeout_s=21600,
    )


def generate_missing() -> None:
    ops = discovered_ops()
    if not ops:
        raise RuntimeError("No profiled ops discovered after unique profiling")
    missing = [op for op in ops if op not in generated_success_ops()]
    if not missing:
        log(f"SKIP generate: all {len(ops)} discovered ops have generated kernels")
        return
    run_logged(
        "generate_unique",
        [
            sys.executable,
            "-m",
            "src.optimizer.workflow",
            "generate",
            "--project",
            PROJECT,
            "--ops",
            ",".join(missing),
            "--benchmark",
            "--target-device",
            "cuda",
            "--llm-model",
            "claude-opus-4-7",
            "--llm-provider",
            "anthropic",
            "--workers",
            "1",
        ],
        timeout_s=86400,
    )


def max_node_id(op: str) -> int:
    db_path = PROJECT_ROOT / "trees" / op / "nodes.db"
    if not db_path.exists():
        return -1
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT MAX(id) FROM nodes").fetchone()
    return int(row[0]) if row and row[0] is not None else -1


def tree_state(ops: list[str]) -> dict[str, int]:
    return {op: max_node_id(op) for op in ops}


def optimize_to(target: int) -> None:
    while True:
        ops = discovered_ops()
        if not ops:
            raise RuntimeError("No profiled ops discovered before optimization")
        state = tree_state(ops)
        update_status(tree_state=state, target=target)
        remaining = [op for op, node_id in state.items() if node_id < target]
        if not remaining:
            log(f"optimize_{target} tree target reached: {state}")
            return
        min_existing = min(state[op] for op in remaining)
        iterations = max(1, target - min_existing)
        run_logged(
            f"optimize_to_{target}",
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
            timeout_s=86400,
        )


def arm_manifest_exists(arm: str) -> bool:
    return any(ARTIFACT_ROOT.glob(f"*__{arm}__*/collection_manifest.json"))


def latest_arm_manifest(arm: str) -> Path:
    manifests = sorted(
        ARTIFACT_ROOT.glob(f"*__{arm}__*/collection_manifest.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not manifests:
        raise FileNotFoundError(f"No collection manifest found for {arm}")
    return manifests[-1]


def collect_opt10() -> None:
    arm = "optimize_10"
    if arm_manifest_exists(arm):
        log(f"SKIP collect {arm}: {latest_arm_manifest(arm)} exists")
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
            arm,
        ],
        timeout_s=3600,
    )


def main() -> int:
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"Project not found: {PROJECT_ROOT}")
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    update_status(
        created_at_utc=read_json(STATUS_PATH, {}).get("created_at_utc", utc_now()),
        driver_pid=os.getpid(),
        protocol={
            "project": PROJECT,
            "model": "stabilityai/stable-diffusion-3.5-medium",
            "quantization": "bf16",
            "target_device": "gb10",
            "profile_capture_mode": "unique",
            "llm_provider": "anthropic",
            "llm_model": "claude-opus-4-7",
            "target": TARGET,
        },
    )
    log("driver started")
    profile_unique()
    generate_missing()
    optimize_to(TARGET)
    collect_opt10()
    update_status(done=True, finished_at_utc=utc_now(), active_step=None)
    log("driver completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
