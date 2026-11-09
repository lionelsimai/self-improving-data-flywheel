"""Synthetic data generator: turns failures into training examples.

Corrections plus deterministic *synonym-swap* paraphrases. No LLM, no API.

Paraphrases change the wording of the question (e.g. "What is" -> "What's",
"largest" -> "biggest"). They deliberately do NOT use prefixes/suffixes that
an evaluator might strip: before 0.1.1 the generator wrapped prompts in
"Please answer: ..." / "... Answer concisely." and the evaluator's
normaliser stripped exactly those strings, so "paraphrase generalisation"
was guaranteed by a shared codebook between generator and grader.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .feedback import Feedback


@dataclass
class TrainingExample:
    """One instruction-tuning example."""

    prompt: str
    completion: str
    source: str  # feedback id / synthetic strategy


# (pattern, replacement) — each matching rule yields one paraphrase.
_SYNONYM_RULES: list[tuple[str, str]] = [
    (r"\bwhat is\b", "what's"),
    (r"\bhow many\b", "what number of"),
    (r"\bwho wrote\b", "who is the author of"),
    (r"\bwho painted\b", "who is the artist behind"),
    (r"\blargest\b", "biggest"),
    (r"\bspoken\b", "used"),
    (r"\bcapital of\b", "capital city of"),
    (r"\bmultiplied by\b", "times"),
    (r"\babsorb\b", "take in"),
]


def _paraphrases(prompt: str) -> list[str]:
    """Return up to 3 synonym-swap paraphrases of ``prompt``."""
    out: list[str] = []
    for pattern, replacement in _SYNONYM_RULES:
        swapped, n = re.subn(pattern, replacement, prompt, count=1,
                             flags=re.IGNORECASE)
        if n and swapped != prompt:
            out.append(swapped)
        if len(out) >= 3:
            break
    return out


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
            out.append(TrainingExample(prompt=para, completion=fb.expected,
                                       source=f"paraphrase-{i}-{j}"))
    return out
