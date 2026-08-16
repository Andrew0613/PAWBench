"""PAWBench: outcome-distribution evaluation of physical-scene video generation.

The public surface is organized into four subpackages, each exposing its
interface from its own ``__init__``:

- ``pawbench.benchmark`` — load and validate a benchmark package.
- ``pawbench.submission`` — load, validate, and build a video submission.
- ``pawbench.paweval`` — run the fixed two-axis VLM judge.
- ``pawbench.results`` — serialize and load public result bundles.
"""

__version__ = "0.1.0"
