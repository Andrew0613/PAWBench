from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "examples" / "quickstart.py"
SPEC = importlib.util.spec_from_file_location("quickstart", SCRIPT)
assert SPEC and SPEC.loader
quickstart = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quickstart
SPEC.loader.exec_module(quickstart)


def test_main_passes_a_persistent_output_directory(monkeypatch, tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    rollouts = tmp_path / "rollouts"
    output = tmp_path / "evaluation"
    benchmark.mkdir()
    rollouts.mkdir()
    (benchmark / "manifest.json").write_text(json.dumps({}), encoding="utf-8")
    scene = rollouts / "A-01"
    scene.mkdir()
    (scene / "r000.mp4").write_bytes(b"video")
    values = {
        "PAWBENCH_DATA_DIR": benchmark,
        "PAWBENCH_RESULTS_DIR": rollouts,
        "PAWBENCH_OUTPUT_DIR": output,
        "PAWBENCH_MODEL": "model-x",
        "PAWBENCH_VLM_BASE_URL": "https://judge.example/v1",
        "PAWBENCH_VLM_MODEL": "judge",
        "PAWBENCH_VLM_API_KEY": "test-key",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))
    captured = {}

    def fake_evaluate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "status": "ok",
            "blockers": [],
            "metrics": {},
            "artifacts": {"rows": str(output / "rows.jsonl")},
        }

    monkeypatch.setattr(quickstart, "evaluate", fake_evaluate)

    quickstart.main()

    assert captured["kwargs"]["output_dir"] == output
    assert captured["kwargs"]["model_or_lane"] == "model-x"
