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

# Evaluate exactly those generated videos.
export OPENAI_API_KEY="..."
python evaluate.py \
  --benchmark "$BENCHMARK_DIR" \
  --videos "$RUN_DIR/rollouts" \
  --output "$RUN_DIR/evaluation" \
  --model "$MODEL_NAME" \
  --vlm-base-url "https://api.openai.com/v1" \
  --vlm-model "your-vlm-model"
```

Generation skips existing videos unless `--overwrite` is supplied. Evaluation
stores checkpoints under `$RUN_DIR/evaluation`, so the same command resumes an
interrupted run. Replace `GENERATOR_MODEL` with another compatible Diffusers
image-to-video checkpoint; no evaluator code changes are required.
