"""Hand-written training/demo feedback seed.

These records are the flywheel's *training* data. They are deliberately
DISJOINT from the held-out eval set in :mod:`data_flywheel.evaluate`:
no prompt (exact or normalised) and no record id appears in both sets,
which the test suite asserts. (Before 0.1.1 the demo seeded its training
data from the held-out set itself — train-on-test — so its +1.00 delta
measured memorisation of the eval questions, nothing else.)
"""
from __future__ import annotations

from .feedback import Feedback

# (id, prompt, wrong response, gold correction)
_ROWS: list[tuple[str, str, str, str]] = [
    ("train-01", "What is the capital of Japan?", "London", "Tokyo"),
    ("train-02", "What is the capital of Australia?", "Sydney", "Canberra"),
    ("train-03", "How many minutes are in one hour?", "100", "60"),
    ("train-04", "What is 3 multiplied by 5?", "20", "15"),
    ("train-05", "What is the chemical symbol for gold?", "Ag", "Au"),
    ("train-06", "Who painted the Mona Lisa?", "Vincent van Gogh",
     "Leonardo da Vinci"),
    ("train-07", "What language is spoken in Japan?", "Mandarin", "Japanese"),
    ("train-08", "What gas do plants absorb from the air?", "Oxygen",
     "Carbon dioxide"),
    ("train-09", "How many sides does a hexagon have?", "8", "6"),
    ("train-10", "What is the square root of 81?", "8", "9"),
]


def default_training_feedback() -> list[Feedback]:
    """Return the hand-written demo seed: 10 thumbs-down + 1 thumbs-up."""
    out = [Feedback(id=rid, prompt=p, response=r, rating=-1, expected=e)
           for rid, p, r, e in _ROWS]
    out.append(Feedback(id="train-up-01", prompt="Say hello",
                        response="Hello!", rating=1))
    return out
