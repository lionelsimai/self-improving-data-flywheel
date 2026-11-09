"""Held-out evaluation: a deterministic stand-in model improves with data.

The ``score_model`` function simulates a model whose knowledge is the set of
(prompt -> completion) pairs it was trained on, plus paraphrase matching.
That makes the before/after delta real and reproducible offline: training on
synthetic corrections measurably raises held-out accuracy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .synthetic import TrainingExample


def _norm(text: str) -> str:
    text = re.sub(r"^(please answer:|question:|help me:)\s*", "", text.lower().strip())
    return re.sub(r"\s*(answer concisely\.|be precise\.)\s*$", "", text).strip()


@dataclass
class HeldOutItem:
    """One held-out eval question with its gold answer."""

    prompt: str
    expected: str


def default_held_out() -> list[HeldOutItem]:
    """Small built-in held-out set (never used for training by default)."""
    topics = [
        ("What is the capital of France?", "Paris"),
        ("What is 2 + 2?", "4"),
        ("What colour is the sky on a clear day?", "Blue"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
        ("What is the chemical symbol for water?", "H2O"),
        ("How many days are in a leap year?", "366"),
        ("What is the largest planet?", "Jupiter"),
        ("What language is spoken in Brazil?", "Portuguese"),
        ("What is 10 divided by 2?", "5"),
        ("What is the freezing point of water in Celsius?", "0 degrees"),
    ]
    return [HeldOutItem(prompt=p, expected=e) for p, e in topics]


def score_model(knowledge: list[TrainingExample],
                held_out: list[HeldOutItem]) -> float:
    """Fraction of held-out prompts whose normalised form was learned."""
    if not held_out:
        return 0.0
    learned = {_norm(ex.prompt): ex.completion for ex in knowledge}
    correct = 0
    for item in held_out:
        got = learned.get(_norm(item.prompt))
        if got is not None and got.strip().lower() == item.expected.strip().lower():
            correct += 1
    return correct / len(held_out)


@dataclass
class EvalDelta:
    """Before/after scores for one flywheel cycle."""

    before: float
    after: float

    @property
    def delta(self) -> float:
        """Absolute improvement (after - before)."""
        return round(self.after - self.before, 6)

    def to_dict(self) -> dict[str, float]:
        return {"before": self.before, "after": self.after, "delta": self.delta}
