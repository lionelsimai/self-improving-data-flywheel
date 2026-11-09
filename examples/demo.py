"""Run one flywheel cycle on seeded demo feedback.

Run: python examples/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_flywheel.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["demo", "--work", "runs/demo"]))
