# PAWBench: How Far Are We from Probabilistically Aligned World Modeling?

<p align="center">
  Paper (arXiv: coming soon) · Project Page (coming soon) ·
  <a href="https://huggingface.co/datasets/Andrew613/PAWBench">Hugging Face Dataset</a> ·
  Leaderboard (coming soon)
</p>

> [!WARNING]
> **Pre-release repository.** The evaluator code and rubric roster are ready for review. The benchmark data are hosted separately on Hugging Face; public release and leaderboard publication remain separate decisions.

## Overview

Physical processes are stochastic: the same action can lead to several valid
outcomes. A realistic video generator should therefore reproduce a *distribution*
of outcomes across repeated samples, rather than optimize for one visually
convincing trajectory.

PAWBench measures that capability with a fixed set of physical scenes. For each
scene, a model generates repeated videos from the same source image and action.
PAWEval reads the generated evidence with a scene-specific outcome rubric and a
trustworthiness audit; the official metrics then compare observed outcomes with
the released reference contract.

| Benchmark component | Released contract |
| --- | --- |
| Scenes | 50 physical scenes: 25 Calibration and 25 Coverage |
| Rollouts | 50 generated videos per scene |
| Calibration target | Reference outcome distribution |
| Coverage target | Set of supported outcome labels |
| PAWEval rubrics | 50 outcome + 50 trustworthiness YAML files |
| Public Python API | `evaluate()` and `compute_metrics()` |

## What PAWBench measures

PAWBench has two complementary tracks.

| Track | Question | Official metric |
| --- | --- | --- |
| **Calibration** | Does the model reproduce how frequently each outcome occurs? | Total variation distance (TVD); lower is better. |
| **Coverage** | Does the model produce every outcome the scene supports? | Support coverage; higher is better. |

The tracks stay separate. PAWBench intentionally does **not** collapse them
into a single leaderboard score: matching a distribution and discovering its
support are different capabilities.

## Evaluation at a glance

```text
local benchmark package + 50 × 50 generated videos
                         │
                         ▼
              PAWEval evidence and rubrics
              ├── outcome readout
              └── trustworthiness audit
                         │
                         ▼
        normalized judgment rows → official metrics
```

`evaluate()` runs this complete local journey. `compute_metrics()` is the
deterministic metric-only entry point for already-normalized judgment rows.
There is no CLI, downloader, model registry, submission protocol, or experiment
framework hidden behind those functions.

## Installation

Clone this repository and install the evaluator from source. Frame extraction
requires the optional `eval` dependencies.

```bash
git clone https://github.com/Andrew0613/PAWBench.git
cd PAWBench

python -m venv .venv
source .venv/bin/activate
pip install -e ".[eval]"
```

The core package depends only on PyYAML. It does not download benchmark data or
contact a provider during installation.

## Benchmark data

PAWBench expects an already-materialized benchmark package. The evaluator does
not prescribe where you store it or download it on your behalf.

```text
PAWBench/
├── manifest.json
├── scenes.jsonl
├── source_images/          # source images referenced by scenes.jsonl
└── prompts/gt_guided/      # optional prompt bank used by guided experiments
```

The released scene table defines 50 scenes. Each row contains a `scene_id`,
the `split` (`calibration` or `coverage`), a `source_image_path`, outcome
labels, and either:

- `reference_distribution` for a Calibration scene; or
- the supported `outcome_labels` for a Coverage scene.

The benchmark package is hosted at
[`Andrew613/PAWBench`](https://huggingface.co/datasets/Andrew613/PAWBench).
It is a clean distribution of the PAWBench V2 input contract used for the
reported evaluations. Materialize it locally before evaluation.

## Quick start: evaluate generated videos

Provide one item per generated video. Each item identifies the scene and its
repeat index, then points to the local MP4. `evaluate()` derives the complete
50-scene × 50-rollout grid from the benchmark package; it never infers a
smaller denominator from the videos that happen to be present.

```python
from pathlib import Path

from pawbench import evaluate

result = evaluate(
    Path("/data/PAWBench"),
    [
        {
            "sample_id": "my-model::A-01::r000",
            "scene_id": "A-01",
            "repeat_index": 0,
            "video_path": "/results/A-01/r000.mp4",
        },
        # Add one row for every scene and repeat in the released grid.
    ],
    model_or_lane="my-model",
    vlm={
        "base_url": "https://your-vlm-provider.example/v1",
        "model": "your-vlm-model",
        "api_key_env": "YOUR_VLM_API_KEY",
    },
)

print(result["status"])
print(result["metrics"])
```

Set the provider key in the named environment variable before running:

```bash
export YOUR_VLM_API_KEY="..."
```

The full editable example is in [`examples/quickstart.py`](examples/quickstart.py).

## Metric-only use

If PAWEval judgment rows have already been produced, compute official metrics
without decoding media or calling a VLM:

```python
from pawbench import compute_metrics

metrics = compute_metrics(rows, scene_policy)
```

For Calibration, PAWBench reports

```text
TVD (%) = 100 × ½ × Σ |observed outcome frequency − reference probability|
```

For Coverage, it reports

```text
Coverage (%) = 100 × |observed labels ∩ supported labels| / |supported labels|
```

Metric output contains separate per-track, per-group, and scene-pass-rate
aggregates. It does not return a combined ranking.

## PAWEval and rubrics

`pawbench/paweval/` is fully visible for review and reproduction, while the
package root deliberately exposes only `evaluate` and `compute_metrics`.

For each video, PAWEval:

1. samples frames and combines them with the scene source image;
2. renders the fixed outcome and trustworthiness instructions with the
   scene-specific rubric;
3. sends the multimodal request to the configured OpenAI-compatible VLM; and
4. parses the two structured responses into a normalized judgment row.

The 100 YAML rubrics are versioned in
[`pawbench/paweval/rubrics`](pawbench/paweval/rubrics): 50 outcome rubrics and
50 trustworthiness rubrics. The outcome readout supplies the official metric
label. The trustworthiness audit is retained alongside each judgment as a
diagnostic record; it is not a second aggregate score.

## Reproducibility and failure semantics

PAWBench is strict about the benchmark denominator.

- Missing, duplicate, malformed, or failed video items become explicit
  infrastructure rows or blockers; they never reduce the 50-rollout target.
- An invalid 50-scene policy or a missing/mismatched rubric fails before media
  extraction or provider calls.
- Provider credentials, raw responses, local media paths, and internal run
  identifiers are not returned in the public result object.
- The result object always contains `status`, `blockers`, `rows`, and `metrics`.

This policy makes a blocked evaluation distinguishable from a lower-scoring,
but complete, model run.

## Repository layout

```text
pawbench/
├── __init__.py             # public API: evaluate, compute_metrics
├── evaluation.py           # local benchmark → PAWEval → metrics
├── metrics.py              # deterministic official metrics
└── paweval/
    ├── adapter.py          # local media → multimodal request payload
    ├── evidence/           # sampling and frame extraction
    ├── judge/              # OpenAI-compatible request, parsing, retry
    ├── judgment.py         # two-axis PAWEval orchestration
    └── rubrics/            # 100 versioned scene rubrics

examples/                   # minimal local workflow
tests/                      # metric, evaluator, rubric, and API checks
```

## Citation

The paper and canonical citation will be added with the public benchmark
release. Until then, please cite the exact repository commit used for an
evaluation rather than treating this pre-release checkout as a published
benchmark version.

## License

PAWBench is released under the [Apache License 2.0](LICENSE).
