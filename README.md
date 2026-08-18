# PAWBench

PAWBench evaluates whether a video model reproduces the outcome distribution
of physical scenes. It compares repeated generated outcomes against a
calibrated reference distribution; it is not a single-video plausibility test.

## Status

**Pre-release.** This repository is ready for an explicit later publication
decision; this status does not publish data, a package, or a release. Its
supported Python surface is:

```python
from pawbench import compute_metrics, evaluate
```

`compute_metrics()` and `evaluate()` are the two public entry points. The
package does not define a custom
benchmark-package format, submission format, result-bundle format, downloader,
command-line interface, model adapter, provider registry, scheduler, or
experiment runtime.

PAWEval is the evaluator implementation used by `evaluate()`. Its source,
prompt templates, and scene-specific rubrics are shipped in
`pawbench/paweval/` for review and reproduction. When evaluation runs, it sends
the source image and sampled frames from the generated video to the configured
VLM provider for judgment. Install its local media decoder with:

```bash
pip install "pawbench[eval]"
```

Provider credentials, raw provider responses, local media paths, and internal
run identifiers are not included in the resulting judgment rows.
The benchmark provides the measured scene and reference distribution; PAWEval
loads one outcome rubric and one trustworthiness rubric by `scene_id`. The
public release contains 50 rubrics on each axis, for 100 files in total. The
two are intentionally separate: the benchmark defines what is evaluated, while
the rubric defines how evidence is judged.

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

The official data location is the package pinned by the release manifest:
`hf://Andrew613/PAWBench-Results@5fdea8a1e7a1e6ccf69bf5af9cf7947aefc58190/benchmark/PAWBench_V2`.
Download or otherwise materialize that package locally before evaluation;
PAWBench itself does not download from Hugging Face.

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

## Evaluation

`evaluate(benchmark_path, videos, model_or_lane=..., vlm=...)` runs the full
local journey. `benchmark_path` is an already-downloaded package with its
`manifest.json` and `scenes.jsonl`. Each element of `videos` names one generated
video with `sample_id`, `scene_id`, `repeat_index`, and `video_path`.

The function derives the full 50 scenes × 50 repeats grid from the benchmark,
judges each supplied video with PAWEval, and returns both metric-ready rows and
the official metrics. Missing, duplicate, malformed, and failed items are
returned as explicit blockers or infrastructure rows; they never shrink the
denominator. `vlm` supplies only `base_url`, `model`, and `api_key_env` (plus
optional timeout and token settings).

The returned JSON-compatible object has `status`, `blockers`, `rows`, and
`metrics`. Each row retains its sample, scene, model/lane, and repeat identity;
it is either an `outcome`, a `null_observation`, or an
`infrastructure_failure`. `metrics` contains separate `calibration` and
`coverage` tracks—there is no combined ranking.

## Quickstart

Create a Python list of generated-video rows, then call the package directly:

```python
from pathlib import Path

from pawbench import evaluate

result = evaluate(
    Path("/data/PAWBench_V2"),
    [
        {
            "sample_id": "my-model::A-01::r000",
            "scene_id": "A-01",
            "repeat_index": 0,
            "video_path": "/results/A-01/r000.mp4",
        },
        # one row for every scene and repeat in the released grid
    ],
    model_or_lane="my-model",
    vlm={
        "base_url": "https://your-vlm-provider.example/v1",
        "model": "your-vlm-model",
        "api_key_env": "YOUR_VLM_API_KEY",
    },
)
print(result["status"], result["metrics"])
```

See [`examples/quickstart.py`](examples/quickstart.py) for the same workflow
as a small editable script. `evaluate()` returns blocked output when the grid
is incomplete or a provider/media problem occurs; it does not silently score a
smaller benchmark.

## Package boundary

```text
pawbench/
├── __init__.py             # exports only evaluate and compute_metrics
├── evaluation.py           # local benchmark → PAWEval → metrics
├── metrics.py              # deterministic official metrics
└── paweval/                # visible evaluator implementation, not a judge SDK
    ├── judgment.py         # two-axis judgment orchestration
    ├── adapter.py          # multimodal request construction
    ├── evidence/           # frame sampling, extraction, and local media package
    ├── judge/              # OpenAI-compatible client, parsing, and retry
    └── rubrics/            # loader, validation, 50 outcome + 50 trust rubrics

examples/           # explains the release-data boundary
tests/              # metric, evaluator, rubric, and public-interface tests
```

PAWEval remains an implementation detail, not a public judge SDK. The public
package has no CLI, downloader, provider registry, experiment runner, run
identity, artifact ledger, or media-hash workflow.

## Installation

```bash
pip install pawbench
```

## License

Apache-2.0. See [LICENSE](LICENSE).
