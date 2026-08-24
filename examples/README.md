# Examples

[`quickstart.py`](quickstart.py) is the complete local workflow: point it at a
downloaded benchmark package and one model's `<scene_id>/r000.mp4` …
`r049.mp4` rollouts, configure the PAWEval VLM with environment variables, and
run it from the repository root. Its exact setup commands are in the
[README](../README.md#3-evaluate-your-videos).

The repository intentionally does not ship a generator, a submission protocol,
or a model registry.
