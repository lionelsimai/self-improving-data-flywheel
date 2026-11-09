"""LoRA fine-tune: config, validation and a no-GPU --dry-run manifest.

Real training (with a GPU + peft/transformers) consumes the same validated
dataset and config; the dry run proves the data pipeline end-to-end.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .synthetic import TrainingExample


@dataclass
class LoraConfig:
    """LoRA hyperparameters (validated before any run)."""

    base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 3
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    def validate(self) -> list[str]:
        """Return a list of config errors (empty = valid)."""
        errs: list[str] = []
        if self.rank <= 0:
            errs.append("rank must be > 0")
        if not (0.0 <= self.dropout < 1.0):
            errs.append("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            errs.append("learning_rate must be > 0")
        if self.epochs <= 0:
            errs.append("epochs must be > 0")
        return errs


def load_lora_config(path: str | Path) -> LoraConfig:
    """Parse and validate ``config/lora.yaml`` into a :class:`LoraConfig`.

    Stdlib-only mini-parser for the flat ``key: value`` file (the project
    has no YAML dependency). Raises ``ValueError`` on a missing key, an
    unparsable value, or a field that fails ``LoraConfig.validate()``.
    """
    raw: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        raw[key.strip()] = value.strip()

    def _req(key: str) -> str:
        if key not in raw:
            raise ValueError(f"lora config {path} is missing key: {key}")
        return raw[key]

    try:
        modules = tuple(m.strip() for m in
                        _req("target_modules").strip("[]").split(",") if m.strip())
        cfg = LoraConfig(
            base_model=_req("base_model"), rank=int(_req("rank")),
            alpha=int(_req("alpha")), dropout=float(_req("dropout")),
            learning_rate=float(_req("learning_rate")),
            epochs=int(_req("epochs")), target_modules=modules)
    except ValueError as e:
        raise ValueError(f"invalid lora config {path}: {e}") from e
    errors = cfg.validate()
    if errors:
        raise ValueError(f"invalid lora config {path}: " + "; ".join(errors))
    return cfg


def dry_run(examples: list[TrainingExample], config: LoraConfig,
            out_dir: str | Path,
            lora_config_path: str | Path | None = None) -> dict[str, object]:
    """Validate data + config and write a run manifest. No GPU, no training.

    When ``lora_config_path`` is given, that YAML file is read and
    validated and *its* fields are recorded in the manifest (the passed
    ``config`` is then only a fallback for when no file is supplied).
    """
    source = "defaults"
    if lora_config_path is not None:
        config = load_lora_config(lora_config_path)
        source = str(lora_config_path)
    errors = config.validate()
    if not examples:
        errors.append("no training examples")
    for i, ex in enumerate(examples):
        if not ex.prompt.strip() or not ex.completion.strip():
            errors.append(f"example {i} has empty prompt/completion")
    if errors:
        raise ValueError("; ".join(errors))
    payload = "\n".join(f"{e.prompt}\t{e.completion}" for e in examples)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    manifest: dict[str, object] = {
        "mode": "dry-run", "examples": len(examples), "dataset_sha": digest,
        "config": {**asdict(config), "target_modules": list(config.target_modules)},
        "lora_config_source": source,
        "status": "validated — ready for GPU training",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(asdict(e)) for e in examples) + "\n", encoding="utf-8")
    return manifest
