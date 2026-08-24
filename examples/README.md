# Examples

[`quickstart.py`](quickstart.py) is the complete local workflow: point it at a
downloaded benchmark package and one model's `<scene_id>/r000.mp4` …
`r049.mp4` rollouts, configure the PAWEval VLM with environment variables, and
run it from the repository root. Its exact setup commands are in the
[README](../README.md#3-evaluate-your-videos).

[`generate_diffusers.py`](generate_diffusers.py) is an optional, editable
generation example. Give it any compatible Diffusers image-to-video model ID;
it reads the benchmark source images and prompts and writes the rollout layout
consumed by `quickstart.py`. It is a local example, not a model registry or
submission system.
