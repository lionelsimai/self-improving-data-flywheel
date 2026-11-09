"""CLI for the data flywheel.

  python -m data_flywheel.cli demo
  python -m data_flywheel.cli collect --db data/feedback.jsonl --prompt Q --response A --rating -1 --expected E
  python -m data_flywheel.cli cycle --db data/feedback.jsonl --work runs/cycle1
  python -m data_flywheel.cli train --data runs/cycle1/run/train.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cycle import run_cycle
from .evaluate import default_held_out
from .feedback import Feedback, FeedbackStore
from .synthetic import TrainingExample
from .train import LoraConfig, dry_run


def _seed_demo_db(path: Path) -> None:
    store = FeedbackStore(path)
    if store.path.exists():
        store.path.unlink()
    for item in default_held_out():
        store.add(Feedback(prompt=item.prompt, response="(wrong answer)",
                           rating=-1, expected=item.expected))
    store.add(Feedback(prompt="Say hello", response="Hello!", rating=1))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="flywheel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--work", default="runs/demo")
    c = sub.add_parser("collect")
    c.add_argument("--db", default="data/feedback.jsonl")
    c.add_argument("--prompt", required=True)
    c.add_argument("--response", required=True)
    c.add_argument("--rating", type=int, required=True)
    c.add_argument("--expected", default="")
    cy = sub.add_parser("cycle")
    cy.add_argument("--db", default="data/feedback.jsonl")
    cy.add_argument("--work", default="runs/cycle1")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--dry-run", action="store_true")
    t.add_argument("--out", default="runs/train")
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        db = Path(args.work) / "feedback.jsonl"
        _seed_demo_db(db)
        report = run_cycle(db, args.work)
        print(json.dumps(report, indent=2, default=str))
        return 0
    if args.cmd == "collect":
        FeedbackStore(args.db).add(Feedback(
            prompt=args.prompt, response=args.response,
            rating=args.rating, expected=args.expected))
        print("Recorded.")
        return 0
    if args.cmd == "cycle":
        print(json.dumps(run_cycle(args.db, args.work), indent=2, default=str))
        return 0
    # train
    rows = [json.loads(line) for line in
            Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = [TrainingExample(**r) for r in rows]
    if not args.dry_run:
        print("Real GPU training is out of scope for this build; use --dry-run.")
        return 2
    print(json.dumps(dry_run(examples, LoraConfig(), args.out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
