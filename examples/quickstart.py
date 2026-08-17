"""Minimal PAWBench evaluation workflow; edit the paths and VLM settings."""

from pathlib import Path

from pawbench import evaluate


benchmark = Path("/data/PAWBench_V2")
videos = [
    {
        "sample_id": "my-model::A-01::r000",
        "scene_id": "A-01",
        "repeat_index": 0,
        "video_path": "/results/A-01/r000.mp4",
    },
    # Add one item for every scene and repeat in the released grid.
]

result = evaluate(
    benchmark,
    videos,
    model_or_lane="my-model",
    vlm={
        "base_url": "https://your-vlm-provider.example/v1",
        "model": "your-vlm-model",
        "api_key_env": "YOUR_VLM_API_KEY",
    },
)
print(result["status"])
