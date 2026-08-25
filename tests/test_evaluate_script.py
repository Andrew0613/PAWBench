from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
assert SPEC and SPEC.loader
evaluate_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate_script
SPEC.loader.exec_module(evaluate_script)


def test_main_passes_a_persistent_output_directory(monkeypatch, tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    rollouts = tmp_path / "rollouts"
    output = tmp_path / "evaluation"
    benchmark.mkdir()
    rollouts.mkdir()
    (benchmark / "manifest.json").write_text(json.dumps({}), encoding="utf-8")
    scene = rollouts / "coin-flip"
    scene.mkdir()
    (scene / "r000.mp4").write_bytes(b"video")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
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

    monkeypatch.setattr(evaluate_script, "evaluate", fake_evaluate)

    status = evaluate_script.main(
        [
            "--benchmark",
            str(benchmark),
            "--videos",
            str(rollouts),
            "--output",
            str(output),
            "--model",
            "model-x",
            "--vlm-base-url",
            "https://judge.example/v1",
            "--vlm-model",
            "judge",
        ]
    )

    assert status == 0
    assert captured["kwargs"]["output_dir"] == output
    assert captured["kwargs"]["model_or_lane"] == "model-x"
    assert captured["kwargs"]["vlm"] == {
        "base_url": "https://judge.example/v1",
        "model": "judge",
        "api_key_env": "OPENAI_API_KEY",
    }
