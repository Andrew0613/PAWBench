# PAWBench

PAWBench evaluates whether a video model reproduces the outcome distribution
of physical scenes. It compares repeated generated outcomes against a
calibrated reference distribution; it is not a single-video plausibility test.

## Status

**Pre-release.** This repository has been contracted to its truthful public
boundary. The released public journey will be:

```python
from pawbench import compute_metrics, evaluate
```

`compute_metrics()` is available now. `evaluate()` remains present as the
second public entry point and will be enabled with the complete evaluation
journey. The package does not currently define a custom
benchmark-package format, submission format, result-bundle format, downloader,
command-line interface, model adapter, provider registry, scheduler, or
experiment runtime.

The PAWEval reference implementation is now included as a private component of
the forthcoming `evaluate()` journey. When that journey runs, it sends the
source image and sampled frames from the generated video to the configured VLM
provider for judgment. Install its local media decoder with:

```bash
pip install "pawbench[eval]"
```

Provider credentials, raw provider responses, local media paths, and internal
run identifiers are not included in the resulting judgment rows.
The released scene policy is also PAWEval's rubric source: its action and
`outcome_labels` are rendered for the judge, so this repository does not ship
a second, divergent set of per-scene rubrics.

## Benchmark data

PAWBench consumes the released benchmark data contract rather than defining a
second one. A benchmark release provides a manifest and a 50-scene table:

```text
manifest.json
scenes.jsonl
├── 25 calibration scenes with reference distributions
└── 25 coverage scenes with supported outcome labels
```

Each scene describes its source image identity, action, prompt, and outcome
ontology. The reference evaluator and metric consume that data directly.

## Deterministic metric

`compute_metrics(rows, scene_policy)` is available now and makes no VLM or
network call. `rows` is a JSON-compatible list of normalized judgments. Every
row has `sample_id`, `model_or_lane`, `track`, `scene_id`, `repeat_index`, and
one of these observations:

- `outcome` with an in-ontology `outcome_label`;
- `null_observation`; or
- `infrastructure_failure` with a `failure_code`.

`scene_policy` names the evaluated model or lane and declares the 25
Calibration and 25 Coverage scenes, each with its group, 50 repeat indices,
and either a reference distribution (Calibration) or supported outcome labels
(Coverage). The function validates this boundary, returns separate per-track
and per-group aggregates plus scene-pass rates, and never creates a combined
ranking. An unresolved infrastructure failure blocks its track instead of
silently shrinking the denominator.

## Package boundary

```text
pawbench/
├── evaluation.py   # high-level evaluation entry point
└── metrics.py      # deterministic metric entry point

examples/           # explains the release-data boundary
tests/              # offline package-contract tests
```

The PAWEval reference implementation and rubric assets will be added behind
`evaluate()` in a later release step. They remain implementation details, not
a public judge SDK.

## Installation

```bash
pip install pawbench
```

## License

Apache-2.0. See [LICENSE](LICENSE).
