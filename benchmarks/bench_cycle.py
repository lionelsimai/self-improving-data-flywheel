"""Benchmark: one flywheel cycle on DISJOINT training / held-out data.

The training seed (training_data.py) shares no question with the held-out
eval set. The deterministic stand-in model can only answer questions it
learned verbatim, so the honest expectations are:

- training-question fit after the cycle: 1.00 (the data flows through), and
- held-out delta: +0.00 (the stand-in cannot generalise to unseen questions).

A nonzero held-out delta in this setup would mean leakage, not learning.
The +5%-per-cycle figure is a TARGET for a real LLM run, not a result here.

Run: python benchmarks/bench_cycle.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_flywheel.cycle import run_cycle  # noqa: E402
from data_flywheel.evaluate import HeldOutItem, score_model  # noqa: E402
from data_flywheel.feedback import FeedbackStore  # noqa: E402
from data_flywheel.synthetic import generate_synthetic  # noqa: E402
from data_flywheel.training_data import default_training_feedback  # noqa: E402


def main() -> int:
    """Run a cycle on the disjoint seed and report fit + held-out delta."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "feedback.jsonl"
        store = FeedbackStore(db)
        for fb in default_training_feedback():
            store.add(fb)
        failures = store.failures()
        synthetic = generate_synthetic(failures)
        report = run_cycle(db, Path(tmp) / "work")
    ev = report["eval"]
    train_items = [HeldOutItem(prompt=f.prompt, expected=f.expected)
                   for f in failures]
    fit = score_model(synthetic, train_items)
    print(f"Feedback: {report['feedback']}  Synthetic examples: "
          f"{report['synthetic_examples']}")
    print(f"Training-question fit after cycle: {fit:.2f} "
          f"(data pipeline works end-to-end)")
    print(f"Held-out eval — before: {ev['before']:.2f}  after: {ev['after']:.2f} "  # type: ignore[index]
          f"delta: {ev['delta']:+.2f}")  # type: ignore[index]
    print("Machinery check only: the stand-in model memorises verbatim "
          "questions and cannot generalise, so the held-out delta is +0.00 "
          "by design. +5%/cycle remains a target for a real-model run.")
    assert fit == 1.0 and ev["delta"] == 0.0  # type: ignore[index]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
