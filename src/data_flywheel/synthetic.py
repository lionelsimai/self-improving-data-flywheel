"""Synthetic data generator: turns failures into training examples.

Templated corrections plus deterministic paraphrases. No LLM, no API.
"""
from __future__ import annotations

from dataclasses import dataclass

from .feedback import Feedback

_PREFIXES = ["Please answer: ", "Question: ", "Help me: ", ""]
_SUFFIXES = ["", " Answer concisely.", " Be precise."]


@dataclass
class TrainingExample:
    """One instruction-tuning example."""

    prompt: str
    completion: str
    source: str  # feedback id / synthetic strategy


def _paraphrases(prompt: str) -> list[str]:
    return [f"{pre}{prompt}{suf}" for pre in _PREFIXES for suf in _SUFFIXES][:4]


def generate_synthetic(failures: list[Feedback]) -> list[TrainingExample]:
    """Generate corrected + paraphrased examples from thumbs-down records.

    Records without an ``expected`` correction are skipped — inventing a
    gold answer would poison training.
    """
    out: list[TrainingExample] = []
    for i, fb in enumerate(failures):
        if not fb.expected:
            continue
        out.append(TrainingExample(prompt=fb.prompt, completion=fb.expected,
                                   source=f"correction-{i}"))
        for j, para in enumerate(_paraphrases(fb.prompt)):
            if para != fb.prompt:
                out.append(TrainingExample(prompt=para, completion=fb.expected,
                                           source=f"paraphrase-{i}-{j}"))
    return out
