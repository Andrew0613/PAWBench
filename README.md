# PAWBench: How Far Are We from Probabilistically Aligned World Modeling?

<p align="center">
  <a href="https://arxiv.org/pdf/2608.27345">
    <img src="https://img.shields.io/badge/PAPER-PDF-555555?style=for-the-badge&logo=readthedocs&logoColor=white" alt="PAWBench paper PDF">
  </a>
  <a href="https://arxiv.org/abs/2608.27345">
    <img src="https://img.shields.io/badge/arXiv-2608.27345-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv: 2608.27345">
  </a>
  <a href="https://pawbench.github.io">
    <img src="https://img.shields.io/badge/WEBPAGE-PAWBench-2F80ED?style=for-the-badge&logo=githubpages&logoColor=white" alt="PAWBench webpage">
  </a>
  <a href="https://huggingface.co/datasets/Andrew613/PAWBench">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97_Dataset-PAWBench-FFD21E?style=for-the-badge" alt="Hugging Face dataset">
  </a>
</p>

## Overview

Physical processes are stochastic: the same action can end in more than one
valid state. PAWBench therefore asks whether a video generator reproduces a
*distribution* of physical outcomes across repeated rollouts, not whether it
can produce one convincing video.

![One plausible rollout can look correct even when repeated rollouts reveal a probabilistically unaligned world.](assets/paper/figure-1.png)

PAWBench fixes the source image and action, samples 50 videos, and evaluates the
resulting distribution of physical outcomes.

| Track | Question | Metric |
| --- | --- | --- |
| **PAW-Calibration** | Are outcome frequencies correct? | TVD (lower is better) |
| **PAW-Coverage** | Are all supported outcomes observed? | Coverage (higher is better) |

The benchmark contains 50 scenes: 25 PAW-Calibration and 25 PAW-Coverage.
It ships 100 reviewable rubrics (one outcome rubric and one trustworthiness
rubric per scene). The two metrics remain separate; PAWBench does not turn
them into a single score.

## Evaluate a new video model

PAWBench has one model-to-result workflow. First, use the benchmark inputs to
generate a complete rollout directory. Then pass that directory to PAWEval,
which reads the videos with the official Gemini 3.5 Flash judge and computes
the two benchmark metrics.

```text
benchmark package -> video generator -> rollout directory -> PAWEval -> metrics.json
```

The generator and the judge are different models. `evaluate.py` never loads or
runs the video generator; it evaluates videos that already exist on disk.

### 1. Install and download the benchmark

```bash
git clone https://github.com/Andrew0613/PAWBench.git
cd PAWBench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export PAWBENCH_DATA_DIR="$PWD/data/PAWBench"
pip install -U "huggingface_hub[cli]"
hf download Andrew613/PAWBench \
  --repo-type dataset \
  --local-dir "$PAWBENCH_DATA_DIR"
```

The downloaded directory contains the scene table, source images, generation
prompts, and scoring policy used by the evaluator.

### 2. Generate or provide the rollouts

For every released scene, generate 50 independent videos from the scene's
source image and generation prompt. Any video generator is supported as long
as it writes the complete 50-scene x 50-rollout grid in this layout:

```text
my-model-rollouts/
├── <scene-id>/
│   ├── r000.mp4
│   ├── ...
│   └── r049.mp4
└── ...
```

You can produce this directory with your own generation system or use one of
the editable examples:

- [Diffusers generation](examples/README.md#generate-and-evaluate-with-diffusers)
  for compatible local or Hugging Face checkpoints.
- [OpenRouter generation](examples/README.md#generate-with-openrouter) for
  hosted image-to-video models.

Both examples write the same rollout contract consumed by `evaluate.py`; they
are optional implementations, not separate benchmark protocols.

### 3. Configure the official PAWEval judge

The official PAWBench results use **Gemini 3.5 Flash** through OpenRouter's
OpenAI-compatible endpoint. This credential belongs to the PAWEval judge, not
to the video generator:

```bash
export OPENROUTER_API_KEY="..."
```

`OPENAI_API_KEY` is not required for the official workflow; the evaluation
command below explicitly tells PAWEval to read `OPENROUTER_API_KEY`.
A local generator such as Diffusers does not need this key during generation.
The optional OpenRouter generator also uses `OPENROUTER_API_KEY`, but generation
and PAWEval judging remain separate API stages.

### 4. Run PAWEval

```bash
export MODEL_NAME="my-model"
export RUN_DIR="$PWD/runs/$MODEL_NAME"

python evaluate.py \
  --benchmark "$PAWBENCH_DATA_DIR" \
  --videos "$PWD/my-model-rollouts" \
  --output "$RUN_DIR/evaluation" \
  --model "$MODEL_NAME" \
  --vlm-base-url "https://openrouter.ai/api/v1" \
  --vlm-model "google/gemini-3.5-flash" \
  --vlm-api-key-env OPENROUTER_API_KEY
```

`--model` is the name recorded in the result files; it does not load the video
generator. Re-running the same benchmark, model name, videos, and judge
configuration resumes completed rollout judgments.

Other OpenAI-compatible multimodal judges can be selected with
`--vlm-base-url`, `--vlm-model`, and `--vlm-api-key-env`. Those runs are useful
for analysis, but they are not directly comparable with the official Gemini
3.5 Flash results.

### 5. Read the result

PAWEval writes `run.json`, `checkpoint.jsonl`, `rows.jsonl`, and `metrics.json`
under `--output`. A complete evaluation has `status: "ok"` and no blockers.
The two track averages remain separate. Inspect the result with the Python
standard library:

```bash
python -m json.tool "$RUN_DIR/evaluation/metrics.json"
```

Calibration reports TVD percentage, where lower is better. Coverage reports
support coverage percentage, where higher is better. Their values are stored
under `tracks.calibration.models.<model>.track_average` and
`tracks.coverage.models.<model>.track_average`. If `status` is `"blocked"`,
resolve the listed missing-video, judge, or infrastructure failures before
reporting the result.

## Visual examples

Each strip below is one generated rollout. PAWBench repeats the same source
image and action 50 times, reads one outcome from every rollout, and evaluates
the resulting distribution rather than judging a single video in isolation.
This published rollout illustrates the readout protocol; a single rollout is
not a model-level result.

### PAW-Calibration: Coin flip

![A generated coin-flip rollout beginning with a coin held over a table and ending with the coin lying heads-up.](assets/examples/coin-flip-rollout.png)

**Action:** flick the coin once. This rollout is read as `heads`. Across 50
rollouts, the Head/Tail frequencies are compared with the scene's reference
distribution using TVD.

More qualitative examples are available on the
[project website](https://pawbench.github.io/#visual-evidence).

## How PAWEval works

![PAWEval maps repeated rollouts to physical outcomes and compares their empirical distribution with the reference.](assets/paweval-overview.png)

PAWEval reads every generated video with the scene's outcome rubric, assigns a
terminal outcome, and aggregates the 50 labels into the distribution scored by
TVD or Coverage. A separate trustworthiness rubric records action, continuity,
and physical-process failures without changing the terminal label or score.
All official results use Gemini 3.5 Flash as the PAWEval judge.

## Benchmark results

![Table 1: Main PAWBench results across PAW-Calibration and PAW-Coverage.](assets/paper/table-1.png)

The table reports the official Gemini 3.5 Flash PAWEval configuration. Results
produced with a different judge should identify that judge and should not be
treated as directly comparable with Table 1.

## Repository layout

```text
evaluate.py                 # evaluate one model's completed rollout directory
pawbench/                   # evaluator implementation and official metrics
└── paweval/                # rubrics, evidence preparation, VLM judgment
examples/                   # optional local and hosted generation examples
assets/                     # README and paper figures
requirements*.txt           # evaluation, generation, and development dependencies
tests/                      # evaluator and script checks
```

## Citation

If you use PAWBench, please cite the paper and record the exact GitHub commit
and Hugging Face dataset revision used for your evaluation:

```bibtex
@article{pu2026pawbench,
  title={PAWBench: How Far Are We from Probabilistically Aligned World Modeling?},
  author={Yuandong Pu and Le Zhuo and Sayak Paul and Gabriel Jorge Menezes and Avram Đorđević and Shiyang Li and Yifan Zhou and Bin Fu and Wenlong Zhang and Junjun He and Yu Qiao and Yihao Liu and Jinbo Xing and Xi Chen},
  journal={arXiv preprint arXiv:2608.27345},
  year={2026},
  eprint={2608.27345},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.27345}
}
```

## License

The evaluator and repository source code are released under the
[Apache License 2.0](LICENSE). Benchmark inputs and media are distributed
separately through the
[Hugging Face dataset](https://huggingface.co/datasets/Andrew613/PAWBench) and
[project website](https://pawbench.github.io/); do not infer that those assets
are covered by this repository's code license. Consult each public surface for
its applicable terms and provenance.
