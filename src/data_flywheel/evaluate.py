"""Held-out evaluation for a deterministic stand-in model.

The ``score_model`` function simulates a model whose knowledge is exactly
the set of (prompt -> completion) pairs it was trained on: a held-out item
scores only if its question was learned verbatim (up to the independent
normalisation in ``_norm``). The stand-in therefore cannot generalise to
unseen questions — a held-out delta above zero is only possible when
training data overlaps the eval questions, which the disjoint training
seed (see :mod:`data_flywheel.training_data`) rules out by construction
and the test suite asserts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .synthetic import TrainingExample


def _norm(text: str) -> str:
    """Normalise a question for exact-match comparison.

    Independent of the synthetic generator: lowercase, keep alphanumeric
    tokens, collapse whitespace. It strips nothing the generator adds —
    the generator (synonym swaps) and this normaliser share no codebook.
    """
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass
class HeldOutItem:
    """One held-out eval question with its gold answer."""

    prompt: str
    expected: str
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = re.sub(r"[^a-z0-9]+", "-", self.prompt.lower()).strip("-")


def default_held_out() -> list[HeldOutItem]:
    """Small built-in held-out set.

    Nothing in the default demo or cycle trains on these items: the demo
    seed lives in :mod:`data_flywheel.training_data` and is disjoint from
    this set by prompt (exact and normalised) and by record id — asserted
    by ``test_training_seed_disjoint_from_held_out``. (Before 0.1.1 the
    demo DID seed training from this set while this docstring claimed the
    opposite; the +1.00 delta it produced was memorisation, not learning.)
    """
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
