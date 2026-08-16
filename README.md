# PAWBench

PAWBench asks whether video and world models reproduce the **outcome
distribution** of physical scenes, rather than whether one generated clip
merely looks plausible. A model is shown a source image and a physical
action; PAWBench scores the distribution of final outcomes across repeats
against a calibrated reference, with a strict trustworthiness gate.

**Status: pre-release (alpha).** Interfaces may change until the first
public benchmark package is published.

## What it does

PAWBench serves four jobs:

1. **Load and validate a benchmark package** from Hugging Face or a local
   directory (scenes, outcome vocabularies, exact calibration reference
   counts, prompt sets).
2. **Load, validate, and build a submission** of generated videos — one
   slot per `(scene, repeat)`, with explicit failure rows so the
   denominator can never shrink silently.
3. **Run PAWEval**, the fixed two-axis VLM judge (`outcome_readout` +
   `trustworthiness_audit`), over any OpenAI-compatible HTTPS endpoint.
4. **Serialize and load public results** (`rows.jsonl` + `summary.json`)
   with deterministic, reproducible output.

Scoring follows the retained V2 semantics: Calibration scenes are scored by
total variation distance from the exact reference distribution; Coverage
scenes by support recovery; per-scene failure is gated on error count, and
model summaries are the unweighted macro over passing scenes. Full-grid
submissions get a `formal` score comparable with the paper; partial grids
get an explicitly flagged `diagnostic` score.

## Installation

```bash
pip install pawbench            # core (validation + judge client)
pip install "pawbench[eval]"    # + video frame extraction
pip install "pawbench[hf]"      # + Hugging Face downloads
pip install "pawbench[all]"     # everything
```

## Quickstart

See [`examples/quickstart.py`](examples/quickstart.py). The example
benchmark and submission under `examples/` are fully synthetic.

```python
import pawbench.benchmark
import pawbench.submission

benchmark = pawbench.benchmark.load("examples/benchmark")
submission = pawbench.submission.load(
    "examples/submission/submission.json", benchmark=benchmark
)
print(len(benchmark.scenes), "scenes,", len(submission.items), "items")
```

*(The quickstart grows with each module as it lands; judge evaluation is
wired up in a later release wave.)*

## Repository layout

```text
pawbench/    benchmark/ submission/ paweval/ results/   # public package
schemas/     JSON Schemas for the four public contracts
examples/    fully synthetic benchmark + submission + quickstart
results/     result-bundle format documentation + synthetic example
tests/       interface-level tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).
