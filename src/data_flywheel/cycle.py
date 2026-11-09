"""Cycle runner: feedback -> synthetic -> dry-run train -> eval delta."""
from __future__ import annotations

from pathlib import Path

from .evaluate import EvalDelta, default_held_out, score_model
from .feedback import FeedbackStore
from .synthetic import TrainingExample, generate_synthetic
from .train import LoraConfig, dry_run


def run_cycle(feedback_path: str | Path, work_dir: str | Path,
              base_knowledge: list[TrainingExample] | None = None) -> dict[str, object]:
    """Run one full flywheel cycle and return a JSON-serialisable report."""
    store = FeedbackStore(feedback_path)
    failures = store.failures()
    synthetic = generate_synthetic(failures)
    held_out = default_held_out()
    base = list(base_knowledge or [])
    before = score_model(base, held_out)
    manifest = dry_run(synthetic, LoraConfig(), Path(work_dir) / "run") if synthetic else {
        "mode": "dry-run", "examples": 0, "status": "no synthetic data — skipped"}
    after = score_model(base + synthetic, held_out)
    delta = EvalDelta(before=before, after=after)
    return {"feedback": store.stats(), "failures_used": len(failures),
            "synthetic_examples": len(synthetic), "train_manifest": manifest,
            "eval": delta.to_dict()}
