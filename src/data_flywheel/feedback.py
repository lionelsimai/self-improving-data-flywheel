"""Feedback collector: thumbs up/down stored as JSONL on disk."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Feedback:
    """One user judgement on one model response."""

    prompt: str
    response: str
    rating: int  # +1 thumbs up, -1 thumbs down
    expected: str = ""
    ts: float = 0.0

    def __post_init__(self) -> None:
        if self.rating not in (1, -1):
            raise ValueError("rating must be +1 or -1")
        if not self.ts:
            self.ts = time.time()


class FeedbackStore:
    """Append-only JSONL store for feedback records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, fb: Feedback) -> None:
        """Append one feedback record."""
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(fb)) + "\n")

    def load(self) -> list[Feedback]:
        """Load all records; returns [] if the file does not exist."""
        if not self.path.exists():
            return []
        out: list[Feedback] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(Feedback(**json.loads(line)))
        return out

    def failures(self) -> list[Feedback]:
        """Only thumbs-down records — the flywheel's fuel."""
        return [f for f in self.load() if f.rating == -1]

    def stats(self) -> dict[str, int]:
        """Counts by rating."""
        rows = self.load()
        return {"total": len(rows),
                "up": sum(1 for r in rows if r.rating == 1),
                "down": sum(1 for r in rows if r.rating == -1)}
