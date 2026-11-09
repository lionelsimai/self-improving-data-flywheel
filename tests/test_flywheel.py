"""Unit + end-to-end tests for the data flywheel."""
import json
from pathlib import Path

import pytest

from data_flywheel.cli import main
from data_flywheel.cycle import run_cycle
from data_flywheel.evaluate import _norm, default_held_out, score_model
from data_flywheel.feedback import Feedback, FeedbackStore
from data_flywheel.synthetic import TrainingExample, generate_synthetic
from data_flywheel.train import LoraConfig, dry_run, load_lora_config
from data_flywheel.training_data import default_training_feedback


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


def test_score_model_counts_verbatim_learned_questions() -> None:
    # Mechanics check of the scorer ONLY: the stand-in answers a question
    # iff it learned that exact question. This is not a generalisation
    # claim — training on the eval set is exactly what the demo must not
    # do (see test_training_seed_disjoint_from_held_out).
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


def test_cycle_e2e_disjoint_training_no_heldout_gain(tmp_path: Path) -> None:
    # Training on the disjoint seed must NOT move the held-out score: the
    # stand-in cannot generalise, so any gain would prove leakage.
    db = tmp_path / "fb.jsonl"
    store = FeedbackStore(db)
    for fb in default_training_feedback():
        store.add(fb)
    report = run_cycle(db, tmp_path / "work")
    ev = report["eval"]  # type: ignore[index]
    assert report["failures_used"] == 10
    assert report["synthetic_examples"] > 10
    assert ev["before"] == 0.0 and ev["after"] == 0.0 and ev["delta"] == 0.0  # type: ignore[index]


def test_cli_demo_e2e(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["demo", "--work", str(tmp_path / "demo")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["feedback"] == {"total": 11, "up": 1, "down": 10}
    assert report["eval"]["after"] == 0.0 and report["eval"]["delta"] == 0.0


def test_training_seed_disjoint_from_held_out() -> None:
    training = default_training_feedback()
    held = default_held_out()
    train_prompts = {_norm(f.prompt) for f in training}
    held_prompts = {_norm(i.prompt) for i in held}
    assert train_prompts.isdisjoint(held_prompts)
    # ... and the same must hold for every generated synthetic example.
    synthetic = generate_synthetic([f for f in training if f.rating == -1])
    assert {_norm(e.prompt) for e in synthetic}.isdisjoint(held_prompts)
    train_ids = {f.id for f in training if f.id}
    held_ids = {i.id for i in held}
    assert train_ids and held_ids and train_ids.isdisjoint(held_ids)


def test_paraphrases_survive_evaluator_normalisation() -> None:
    fails = [Feedback(prompt="What is the capital of Japan?", response="x",
                      rating=-1, expected="Tokyo")]
    out = generate_synthetic(fails)
    paras = [e for e in out if e.source.startswith("paraphrase")]
    assert paras, "expected synonym-swap paraphrases"
    # The evaluator's _norm must NOT collapse a paraphrase back onto the
    # original (the pre-0.1.1 prefix/suffix codebook did exactly that).
    assert all(_norm(e.prompt) != _norm(fails[0].prompt) for e in paras)


def test_feedback_load_skips_malformed_and_dedupes(tmp_path: Path) -> None:
    path = tmp_path / "fb.jsonl"
    good = json.dumps({"prompt": "q", "response": "bad", "rating": -1,
                       "expected": "good", "ts": 1.0, "id": "a"})
    path.write_text("\n".join([good, "{not json", good,
                               '{"prompt": "q2", "rating": 7}',
                               '{"prompt": "q3", "response": "ok", "rating": 1}'])
                    + "\n", encoding="utf-8")
    store = FeedbackStore(path)
    rows = store.load()
    assert store.skipped_lines == 2
    assert [r.prompt for r in rows] == ["q", "q3"]  # duplicate (q, good) dropped


def test_dry_run_reads_and_validates_lora_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "lora.yaml"
    yaml_path.write_text(
        "base_model: org/tiny-model\nrank: 4\nalpha: 8\ndropout: 0.1\n"
        "learning_rate: 0.001\nepochs: 2\ntarget_modules: [q_proj]\n",
        encoding="utf-8")
    ex = [TrainingExample(prompt="q", completion="a", source="t")]
    manifest = dry_run(ex, LoraConfig(), tmp_path / "out",
                       lora_config_path=yaml_path)
    cfg = manifest["config"]  # type: ignore[index]
    assert cfg["base_model"] == "org/tiny-model" and cfg["rank"] == 4  # type: ignore[index]
    assert cfg["target_modules"] == ["q_proj"]  # type: ignore[index]
    assert manifest["lora_config_source"] == str(yaml_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("base_model: x\nrank: 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_lora_config(bad)


def test_config_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from data_flywheel.config import Settings
    monkeypatch.setenv("FLYWHEEL_LOG_LEVEL", "DEBUG")
    assert Settings.from_env().log_level == "DEBUG"
