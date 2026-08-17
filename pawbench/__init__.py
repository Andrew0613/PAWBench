"""PAWBench public evaluation surface."""

from pawbench.evaluation import evaluate
from pawbench.metrics import compute_metrics

__version__ = "0.1.0"

__all__ = ["compute_metrics", "evaluate"]
