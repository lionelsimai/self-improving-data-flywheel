# Changelog

All notable changes to this project will be documented in this file.
Format: Keep a Changelog. Versioning: Semantic Versioning.

## [0.1.1] - 2026-11-09
### Fixed
- Train-on-test removed: the demo seeded its training data from the held-out eval set itself, so the reported +1.00 delta was memorisation. The demo now seeds from a new hand-written training set (`training_data.py`) that is disjoint from the held-out set by normalised question and by record id — asserted by `test_training_seed_disjoint_from_held_out`. Honest result on the verbatim stand-in model: held-out delta +0.00 (was +1.00); synthetic examples per demo cycle 23 (was 50).
- False docstring corrected: `default_held_out()` claimed the set was "never used for training" while the demo trained on it.
- Shared generator/grader codebook broken: paraphrases are now synonym swaps, and the evaluator's `_norm` is an independent tokeniser that strips nothing the generator adds (tested: paraphrases no longer normalise back onto the original).
- `config/lora.yaml` is no longer decorative: `train --dry-run` and the cycle read, validate and record its fields in the manifest (`lora_config_source`).
- `FeedbackStore.load()` no longer crashes on a malformed line: bad lines are skipped and counted in `skipped_lines`; records are deduplicated by (prompt, correction).
### Changed
- README: "+5% quality per cycle" is labelled a target for a real-model run, not a result; the stand-in delta is presented as a machinery check. Benchmark now reports training-question fit (1.00) alongside the held-out delta (+0.00) and asserts both.

## [0.1.0] - 2026-11-09
### Added
- Initial public release: working offline demo, CLI, tests, benchmarks, CI gate.
- README with architecture diagram, benchmark section and honest limitations.
