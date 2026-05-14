from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusion3Pipeline


MODEL_ID = os.environ.get(
    "KFORGE_SD35_MODEL_ID",
    "stabilityai/stable-diffusion-3.5-medium",
)
LOCAL_MODEL_DIR = os.environ.get(
    "KFORGE_SD35_DIR",
    "/home/gb10/model-cache/stable-diffusion-3.5-medium",
)
DTYPE = torch.bfloat16
MAX_PROMPTS = int(os.environ.get("KFORGE_VALIDATION_MAX_PROMPTS", "4"))
PROMPT_OFFSET = int(os.environ.get("KFORGE_VALIDATION_PROMPT_OFFSET", "0"))
DEFAULT_HEIGHT = int(os.environ.get("KFORGE_SD35_HEIGHT", "1024"))
DEFAULT_WIDTH = int(os.environ.get("KFORGE_SD35_WIDTH", "1024"))
DEFAULT_STEPS = int(os.environ.get("KFORGE_SD35_STEPS", "28"))
DEFAULT_GUIDANCE = float(os.environ.get("KFORGE_SD35_GUIDANCE_SCALE", "3.5"))
DEFAULT_MAX_SEQUENCE_LENGTH = int(os.environ.get("KFORGE_SD35_MAX_SEQUENCE_LENGTH", "256"))
DEFAULT_OUTPUT_TYPE = os.environ.get("KFORGE_SD35_OUTPUT_TYPE", "pt")

DEFAULT_PROMPTS = [
    "A studio product photo of a red ceramic mug on a walnut table.",
    "A detailed watercolor painting of a mountain village at sunrise.",
    "A small robot reading a newspaper beside a window in soft daylight.",
    "A clean vector poster showing two bicycles under a rainy city streetlight.",
]


def _truthy_env(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _local_snapshot_ready(local: Path) -> bool:
    if not local.exists():
        return False
    if not (local / "model_index.json").exists():
        return False
    expected_dirs = ("transformer", "scheduler", "vae", "tokenizer", "tokenizer_2")
    return all((local / item).exists() for item in expected_dirs)


def _model_source() -> str:
    local = Path(LOCAL_MODEL_DIR).expanduser()
    if _local_snapshot_ready(local):
        return str(local)
    if not _truthy_env("KFORGE_ALLOW_HF_DOWNLOAD", "0"):
        raise FileNotFoundError(
            f"Stable Diffusion 3.5 snapshot is not ready at {local}. "
            "Accept the Stability AI gate on Hugging Face, provide HF_TOKEN, "
            "and set KFORGE_ALLOW_HF_DOWNLOAD=1, or set KFORGE_SD35_DIR to a "
            "complete local diffusers snapshot."
        )
    return MODEL_ID


def _target_device(default: str = "cuda") -> str:
    return os.environ.get("KFORGE_TARGET_DEVICE", default).strip().lower()


class SD35ForgeModel(torch.nn.Module):
    def __init__(self, pipe: StableDiffusion3Pipeline):
        super().__init__()
        self.pipe = pipe
        self.pipe.set_progress_bar_config(disable=True)

    def to(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        device = args[0] if args else kwargs.get("device")
        if device is not None and not _truthy_env("KFORGE_SD35_CPU_OFFLOAD", "0"):
            self.pipe.to(device)
        return self

    def eval(self):  # type: ignore[override]
        for component_name in ("transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
            component = getattr(self.pipe, component_name, None)
            if hasattr(component, "eval"):
                component.eval()
        return self

    def enable_torch_compile(self) -> None:
        torch.set_float32_matmul_precision("high")
        if hasattr(self.pipe, "transformer") and self.pipe.transformer is not None:
            self.pipe.transformer.to(memory_format=torch.channels_last)
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                mode=os.environ.get("KFORGE_SD35_COMPILE_MODE", "max-autotune"),
                fullgraph=True,
            )
        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            self.pipe.vae.to(memory_format=torch.channels_last)
            self.pipe.vae.decode = torch.compile(
                self.pipe.vae.decode,
                mode=os.environ.get("KFORGE_SD35_COMPILE_MODE", "max-autotune"),
                fullgraph=True,
            )

    @torch.no_grad()
    def forward(
        self,
        prompt: str | list[str],
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        seed: int = 1,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        output_type: str = DEFAULT_OUTPUT_TYPE,
    ):
        device = _target_device()
        generator_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        result = self.pipe(
            prompt=prompt,
            negative_prompt="",
            height=int(height),
            width=int(width),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            max_sequence_length=int(max_sequence_length),
            generator=generator,
            output_type=output_type,
        )
        images = getattr(result, "images", result)
        if torch.is_tensor(images):
            return images
        if isinstance(images, (list, tuple)) and images and torch.is_tensor(images[0]):
            return torch.stack(list(images), dim=0)
        return images


def _from_pretrained_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": DTYPE,
        "local_files_only": not _truthy_env("KFORGE_ALLOW_HF_DOWNLOAD", "0"),
    }
    if _truthy_env("KFORGE_SD35_DISABLE_T5", "0"):
        kwargs["text_encoder_3"] = None
        kwargs["tokenizer_3"] = None
    if _truthy_env("KFORGE_SD35_LOW_CPU_MEM_USAGE", "1"):
        kwargs["low_cpu_mem_usage"] = True
    return kwargs


def _build_pipeline() -> StableDiffusion3Pipeline:
    source = _model_source()
    pipe = StableDiffusion3Pipeline.from_pretrained(source, **_from_pretrained_kwargs())
    if _truthy_env("KFORGE_SD35_CPU_OFFLOAD", "0"):
        pipe.enable_model_cpu_offload()
    return pipe


def build_model():
    return SD35ForgeModel(_build_pipeline())


def load_weights(weights_path: str, device: str = "cpu"):
    _ = weights_path
    model = build_model()
    if device and not _truthy_env("KFORGE_SD35_CPU_OFFLOAD", "0"):
        model.to(device)
    return model


def _read_prompt_rows(validation_path: str | None) -> list[dict[str, Any]]:
    if not validation_path:
        return [{"prompt": prompt, "bucket": "default"} for prompt in DEFAULT_PROMPTS]

    root = Path(validation_path)
    candidates = [
        root / "prompts.jsonl",
        root / "texts.jsonl",
        root / "prompts.txt",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".txt":
            rows = [{"prompt": line.strip(), "bucket": "txt"} for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
            if rows:
                return rows
            continue
        rows = []
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or row.get("text") or "").strip()
            if prompt:
                rows.append({**row, "prompt": prompt})
        if rows:
            return rows
    return [{"prompt": prompt, "bucket": "default"} for prompt in DEFAULT_PROMPTS]


def _selected_rows(validation_path: str | None) -> list[dict[str, Any]]:
    rows = _read_prompt_rows(validation_path)
    bucket_filter_raw = os.environ.get("KFORGE_VALIDATION_BUCKETS", "").strip()
    if bucket_filter_raw:
        allowed = {bucket.strip() for bucket in bucket_filter_raw.split(",") if bucket.strip()}
        rows = [row for row in rows if str(row.get("bucket") or "") in allowed]
    return rows[max(0, PROMPT_OFFSET): max(0, PROMPT_OFFSET) + MAX_PROMPTS]


def get_validation_dataloader(validation_path: str | None = None):
    batches = []
    for index, row in enumerate(_selected_rows(validation_path)):
        batches.append(
            {
                "prompt": str(row["prompt"]),
                "height": int(row.get("height") or DEFAULT_HEIGHT),
                "width": int(row.get("width") or DEFAULT_WIDTH),
                "num_inference_steps": int(row.get("num_inference_steps") or row.get("steps") or DEFAULT_STEPS),
                "guidance_scale": float(row.get("guidance_scale") or DEFAULT_GUIDANCE),
                "seed": int(row.get("seed") or index + 1),
                "max_sequence_length": int(row.get("max_sequence_length") or DEFAULT_MAX_SEQUENCE_LENGTH),
                "output_type": str(row.get("output_type") or DEFAULT_OUTPUT_TYPE),
            }
        )
    return batches
