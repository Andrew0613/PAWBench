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

`compute_metrics()` and `evaluate()` are intentionally present as the only
public entry points, but are not available until the corresponding reference
implementations are released. The package does not currently define a custom
benchmark-package format, submission format, result-bundle format, downloader,
command-line interface, model adapter, provider registry, scheduler, or
experiment runtime.

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
