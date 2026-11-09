"""Unit + end-to-end tests for the data flywheel."""
import json
from pathlib import Path

import pytest

from data_flywheel.cli import main
from data_flywheel.cycle import run_cycle
from data_flywheel.evaluate import default_held_out, score_model
from data_flywheel.feedback import Feedback, FeedbackStore
from data_flywheel.synthetic import TrainingExample, generate_synthetic
from data_flywheel.train import LoraConfig, dry_run


def test_feedback_validation() -> None:
    with pytest.raises(ValueError):
        Feedback(prompt="q", response="a", rating=0)


def test_store_roundtrip_and_failures(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.add(Feedback(prompt="q1", response="bad", rating=-1, expected="good"))
    store.add(Feedback(prompt="q2", response="ok", rating=1))
    assert store.stats() == {"total": 2, "up": 1, "down": 1}
    assert len(store.failures()) == 1


def test_synthetic_skips_without_expected() -> None:
    fails = [Feedback(prompt="q", response="bad", rating=-1)]
    assert generate_synthetic(fails) == []


def test_synthetic_generates_correction_and_paraphrases() -> None:
    fails = [Feedback(prompt="What is 2 + 2?", response="5", rating=-1, expected="4")]
    out = generate_synthetic(fails)
    assert len(out) >= 2 and out[0].completion == "4"


def test_score_model_improves_with_training() -> None:
    held = default_held_out()
    assert score_model([], held) == 0.0
    knowledge = [TrainingExample(prompt=i.prompt, completion=i.expected, source="t")
                 for i in held]
    assert score_model(knowledge, held) == 1.0


def test_lora_config_validation() -> None:
    assert LoraConfig().validate() == []
    assert LoraConfig(rank=0).validate() != []


def test_dry_run_writes_manifest(tmp_path: Path) -> None:
    ex = [TrainingExample(prompt="q", completion="a", source="t")]
    manifest = dry_run(ex, LoraConfig(), tmp_path)
    assert manifest["examples"] == 1
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "train.jsonl").exists()


def test_dry_run_rejects_empty() -> None:
    with pytest.raises(ValueError):
        dry_run([], LoraConfig(), Path("/tmp/unused-flywheel"))


def test_cycle_e2e_improves_eval(tmp_path: Path) -> None:
    db = tmp_path / "fb.jsonl"
    store = FeedbackStore(db)
    for item in default_held_out()[:5]:
        store.add(Feedback(prompt=item.prompt, response="wrong",
                           rating=-1, expected=item.expected))
    report = run_cycle(db, tmp_path / "work")
    ev = report["eval"]  # type: ignore[index]
    assert ev["before"] == 0.0 and ev["after"] == 0.5 and ev["delta"] == 0.5  # type: ignore[index]


def test_cli_demo_e2e(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["demo", "--work", str(tmp_path / "demo")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["eval"]["after"] == 1.0 and report["eval"]["delta"] == 1.0


def test_config_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from data_flywheel.config import Settings
    monkeypatch.setenv("FLYWHEEL_LOG_LEVEL", "DEBUG")
    assert Settings.from_env().log_level == "DEBUG"
