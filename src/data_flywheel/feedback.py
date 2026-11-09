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
    id: str = ""

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
        self.skipped_lines = 0

    def add(self, fb: Feedback) -> None:
        """Append one feedback record."""
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(fb)) + "\n")

    def load(self) -> list[Feedback]:
        """Load all records; returns [] if the file does not exist.

        Malformed lines (bad JSON, wrong fields, invalid rating) are
        skipped and counted in ``self.skipped_lines`` instead of crashing
        the load. Records are deduplicated by (prompt, correction) —
        the first occurrence wins.
        """
        self.skipped_lines = 0
        if not self.path.exists():
            return []
        out: list[Feedback] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(Feedback(**json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError):
                self.skipped_lines += 1
        seen: set[tuple[str, str]] = set()
        deduped: list[Feedback] = []
        for fb in out:
            key = (fb.prompt, fb.expected)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fb)
        return deduped

    def failures(self) -> list[Feedback]:
        """Only thumbs-down records — the flywheel's fuel."""
        return [f for f in self.load() if f.rating == -1]

    def stats(self) -> dict[str, int]:
        """Counts by rating."""
        rows = self.load()
        return {"total": len(rows),
                "up": sum(1 for r in rows if r.rating == 1),
                "down": sum(1 for r in rows if r.rating == -1)}
