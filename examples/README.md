# Generation examples

Both examples in this directory read the benchmark source images and prompts,
then write the rollout layout consumed directly by
[`evaluate.py`](../evaluate.py):

```text
<run-dir>/rollouts/<scene-id>/r000.mp4 ... r049.mp4
```

Choose the example that matches where your video model runs:

| Generator | Use it for | Generation credential |
| --- | --- | --- |
| [`generate_diffusers.py`](generate_diffusers.py) | A compatible local or Hugging Face Diffusers checkpoint | None |
| [`generate_openrouter.py`](generate_openrouter.py) | A hosted image-to-video model available through OpenRouter | `OPENROUTER_API_KEY` |

These are editable adapters, not separate benchmark protocols. Both produce
the same 50-scene x 50-rollout input for PAWEval.

Run the commands below from the PAWBench repository root after downloading the
benchmark as described in the [main README](../README.md#1-install-and-download-the-benchmark).

## Generate and evaluate with Diffusers

Install the optional generation dependencies and select a compatible
image-to-video checkpoint:

```bash
pip install -r requirements.txt

BENCHMARK_DIR="$PWD/data/PAWBench"
RUN_DIR="$PWD/runs/wan21"
GENERATOR_MODEL="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
MODEL_NAME="wan21"

python examples/generate_diffusers.py \
  --benchmark "$BENCHMARK_DIR" \
  --output "$RUN_DIR/rollouts" \
  --model-id "$GENERATOR_MODEL" \
  --all-scenes \
  --num-rollouts 50
```

Replace `GENERATOR_MODEL` with another compatible Diffusers checkpoint and
adjust its runtime requirements as needed. The command skips existing videos;
use `--overwrite` only when you intend to regenerate them.

When generation finishes, continue to [Evaluate the rollouts](#evaluate-the-rollouts).

## Generate with OpenRouter

This route uses OpenRouter's asynchronous video API. Each source image must be
available at a directly downloadable public HTTPS URL. The script uses the
public PAWBench Hugging Face dataset by default; if the dataset is not public,
host the same relative files elsewhere and pass that root with
`--source-image-base-url`.

Set the run and model names. The example model supports first-frame
image-to-video at the settings below, but OpenRouter availability and
capabilities can change:

```bash
BENCHMARK_DIR="$PWD/data/PAWBench"
RUN_DIR="$PWD/runs/openrouter-veo"
OPENROUTER_MODEL="google/veo-3.1-lite"
MODEL_NAME="openrouter-veo"
```

First preview the complete 50 x 50 request plan. Preview mode makes no API
calls and spends no credits:

```bash
python examples/generate_openrouter.py \
  --benchmark "$BENCHMARK_DIR" \
  --output "$RUN_DIR/rollouts" \
  --model-id "$OPENROUTER_MODEL" \
  --all-scenes \
  --num-rollouts 50 \
  --duration 4 \
  --resolution 720p \
  --aspect-ratio 16:9
```

Before execution, confirm the selected model's current capabilities with
OpenRouter's video-model endpoint:

```bash
export OPENROUTER_API_KEY="..."

curl "https://openrouter.ai/api/v1/videos/models" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

Then rerun the generation command with `--execute`:

```bash
python examples/generate_openrouter.py \
  --benchmark "$BENCHMARK_DIR" \
  --output "$RUN_DIR/rollouts" \
  --model-id "$OPENROUTER_MODEL" \
  --all-scenes \
  --num-rollouts 50 \
  --duration 4 \
  --resolution 720p \
  --aspect-ratio 16:9 \
  --execute
```

`--execute` can submit 2,500 paid jobs. Job checkpoints are stored under
`$RUN_DIR/rollouts/.openrouter-jobs/`; rerunning with the same options resumes
unfinished jobs and skips downloaded videos. Use only one generator process
per output directory.

When generation finishes, continue to [Evaluate the rollouts](#evaluate-the-rollouts).

## Evaluate the rollouts

Evaluation is identical for both generation routes. The official PAWEval judge
is Gemini 3.5 Flash through OpenRouter:

```bash
export OPENROUTER_API_KEY="..."

python evaluate.py \
  --benchmark "$BENCHMARK_DIR" \
  --videos "$RUN_DIR/rollouts" \
  --output "$RUN_DIR/evaluation" \
  --model "$MODEL_NAME" \
  --vlm-base-url "https://openrouter.ai/api/v1" \
  --vlm-model "google/gemini-3.5-flash" \
  --vlm-api-key-env OPENROUTER_API_KEY
```

The video generator and the PAWEval judge are separate models. A Diffusers run
needs `OPENROUTER_API_KEY` only at evaluation time; an OpenRouter generation
run uses it once for video generation and again for judging. Repeating the
evaluation command resumes completed judgments under `$RUN_DIR/evaluation`.

Inspect the final metrics with:

```bash
python -m json.tool "$RUN_DIR/evaluation/metrics.json"
```
