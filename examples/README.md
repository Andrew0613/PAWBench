# Examples

[`generate_diffusers.py`](generate_diffusers.py) is an optional, editable
generation example. Give it any compatible Diffusers image-to-video model ID;
it reads the benchmark source images and prompts and writes the rollout layout
consumed by [`evaluate.py`](../evaluate.py). It is a local example, not a model
registry or submission system.

## Generate and evaluate with Diffusers

The two existing scripts form one continuous workflow. The rollout directory
written by `generate_diffusers.py` is passed unchanged to `evaluate.py`:

```bash
# From the PAWBench repository root.
pip install -r requirements-generate.txt

BENCHMARK_DIR="$PWD/data/PAWBench"
RUN_DIR="$PWD/runs/wan21"
GENERATOR_MODEL="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
MODEL_NAME="wan21"

# Generate the official 50-scene x 50-rollout evaluation grid.
python examples/generate_diffusers.py \
  --benchmark "$BENCHMARK_DIR" \
  --output "$RUN_DIR/rollouts" \
  --model-id "$GENERATOR_MODEL" \
  --all-scenes \
  --num-rollouts 50

# Evaluate exactly those generated videos with the official PAWEval judge.
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

Generation skips existing videos unless `--overwrite` is supplied. Evaluation
stores checkpoints under `$RUN_DIR/evaluation`, so the same command resumes an
interrupted run. Replace `GENERATOR_MODEL` with another compatible Diffusers
image-to-video checkpoint; no evaluator code changes are required.

## Generate with OpenRouter

[`generate_openrouter.py`](generate_openrouter.py) uses OpenRouter's asynchronous
video API to run a hosted image-to-video model. It writes the same
`<scene-id>/rNNN.mp4` layout as the Diffusers example, so its output can be
passed directly to `evaluate.py`.

OpenRouter requires every source image to be available at a public HTTPS URL.
By default, the script maps the benchmark's relative `source_image_path` onto
the public Hugging Face dataset URL. Before the dataset is public, host the same
relative files at an accessible HTTPS location and pass its root with
`--source-image-base-url`. Do not put credentials in that URL.

The command previews the complete PAWBench generation plan without making an
API call:

```bash
# From the PAWBench repository root.
BENCHMARK_DIR="$PWD/data/PAWBench"
RUN_DIR="$PWD/runs/openrouter"
OPENROUTER_MODEL="google/veo-3.1-lite"

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

Review the plan and current model capabilities, then add `--execute` to submit
the 2,500 paid generation jobs:

```bash
export OPENROUTER_API_KEY="..."

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

OpenRouter model capabilities and availability can change. List the current
models with:

```bash
curl "https://openrouter.ai/api/v1/videos/models" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

The execution command submits 2,500 paid video jobs. Submitted job IDs are
stored under `$RUN_DIR/rollouts/.openrouter-jobs/`; rerunning the command
resumes unfinished jobs only when their request settings still match, and
skips downloaded videos unless `--overwrite` is supplied. Run only one
generator process per output directory.

Official PAWEval judging also uses `OPENROUTER_API_KEY`, with
`google/gemini-3.5-flash` as the judge shown in the workflow above. Video
generation and PAWEval judging are separate API stages even when they use the
same OpenRouter account.
