"""Select bounded source-image and video-frame evidence for PAWEval."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping
from pathlib import Path
from typing import Any


FRAME_COUNT = 8
MAX_IMAGE_EDGE = 768


class MediaFailure(ValueError):
    """Local evidence could not be selected or decoded."""


def build_messages(prompt: str, item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the OpenAI-compatible multimodal content for one judgment axis."""

    source_path = _path(item, "source_image_path")
    video_path = _path(item, "video_path")
    images = [_encode_image(source_path), *_sample_video(video_path)]
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image}})
    return content


def _path(item: Mapping[str, Any], key: str) -> Path:
    value = item.get(key)
    if not isinstance(value, (str, Path)) or not Path(value).is_file():
        raise MediaFailure(f"missing_{key}")
    return Path(value)


def _encode_image(path: Path) -> str:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise MediaFailure("missing_media_dependency") from exc
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise MediaFailure("undecodable_image")
    return _as_data_url(image)


def _sample_video(path: Path) -> list[str]:
    try:
        import cv2
    except ImportError as exc:
        raise MediaFailure("missing_media_dependency") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or total < 1:
            raise MediaFailure("undecodable_video")
        frames = []
        for index in range(FRAME_COUNT):
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(index * (total - 1) / (FRAME_COUNT - 1)))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise MediaFailure("undecodable_video")
            frames.append(_as_data_url(frame))
        return frames
    finally:
        capture.release()


def _as_data_url(image: Any) -> str:
    import cv2

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > MAX_IMAGE_EDGE:
        scale = MAX_IMAGE_EDGE / longest
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise MediaFailure("undecodable_media")
    return "data:image/jpeg;base64," + b64encode(encoded.tobytes()).decode("ascii")
