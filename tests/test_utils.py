"""图片工具与统计工具测试。"""

import numpy as np
import pytest
import supervision as sv

from src.utils import (
    SUPPORTED_EXTENSIONS,
    decode_image_bytes,
    encode_png_bytes,
    summarize_detections,
    validate_upload,
)


def test_supported_extensions_are_jpg_jpeg_png():
    assert SUPPORTED_EXTENSIONS == {".jpg", ".jpeg", ".png"}


def test_validate_upload_rejects_unsupported_format():
    with pytest.raises(ValueError) as excinfo:
        validate_upload("sample.gif", 1024)

    assert "格式" in str(excinfo.value)


def test_validate_upload_rejects_empty_file():
    with pytest.raises(ValueError):
        validate_upload("sample.jpg", 0)


def test_validate_upload_rejects_oversized_file():
    with pytest.raises(ValueError):
        validate_upload("sample.jpg", 30 * 1024 * 1024)


def test_validate_upload_accepts_valid_file():
    assert validate_upload("sample.JPG", 1024) is None


def test_decode_image_bytes_rejects_corrupt_payload():
    with pytest.raises(ValueError):
        decode_image_bytes(b"not-an-image")


def test_decode_image_bytes_decodes_png():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :, 1] = 200

    decoded = decode_image_bytes(encode_png_bytes(image))

    assert decoded.shape == (40, 60, 3)
    assert decoded[0, 0, 1] == 200


def test_summarize_detections_returns_total_and_class_distribution():
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
        confidence=np.array([0.9, 0.4], dtype=np.float32),
        class_id=np.array([0, 0]),
    )

    summary = summarize_detections(detections, class_names=["龟裂", "夹杂"])

    assert summary["total"] == 2
    assert summary["class_counts"] == {"龟裂": 2}
    assert summary["mean_confidence"] == pytest.approx(0.65)


def test_summarize_detections_handles_empty_input():
    summary = summarize_detections(sv.Detections.empty(), class_names=["龟裂"])

    assert summary["total"] == 0
    assert summary["class_counts"] == {}
    assert summary["mean_confidence"] == 0.0
