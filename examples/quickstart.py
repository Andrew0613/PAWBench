"""PAWBench quickstart.

Wave 1 status: package-import smoke only. The quickstart grows with each
release wave:

- wave 2: ``pawbench.benchmark.load()`` over ``examples/benchmark``
- wave 3: ``pawbench.submission.load()`` / ``build()``
- wave 6: ``pawbench.paweval.evaluate()`` and ``pawbench.results.write()``

Run from the repository root:

    python examples/quickstart.py
"""

import sys
from pathlib import Path

# Allow running from a fresh checkout before `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pawbench


def main() -> None:
    print(f"pawbench {pawbench.__version__} imported OK")


if __name__ == "__main__":
    main()
