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


def dry_run(examples: list[TrainingExample], config: LoraConfig,
            out_dir: str | Path) -> dict[str, object]:
    """Validate data + config and write a run manifest. No GPU, no training."""
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
        "status": "validated — ready for GPU training",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(asdict(e)) for e in examples) + "\n", encoding="utf-8")
    return manifest
