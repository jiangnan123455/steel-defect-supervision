"""YOLO 推理封装测试。"""

import numpy as np
import pytest
import supervision as sv

from src.detection import (
    DEFAULT_CLASS_NAMES,
    DetectionError,
    ModelNotFoundError,
    detect_image,
    load_model,
)


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.id = None


class _FakeResult:
    def __init__(self, boxes, names):
        self.boxes = _FakeBoxes(*boxes)
        self.names = names
        self.obb = None
        self.masks = None
        self.orig_shape = (120, 120)


class _FakeModel:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def __call__(self, image, conf=None, verbose=False, **kwargs):
        self.calls.append({"image": image, "conf": conf})
        return self._results


def test_default_class_names_cover_six_neu_defect_types():
    assert len(DEFAULT_CLASS_NAMES) == 6
    assert "龟裂" in DEFAULT_CLASS_NAMES
    assert "划痕" in DEFAULT_CLASS_NAMES


def test_load_model_raises_clear_error_when_weights_missing(tmp_path):
    missing = tmp_path / "best.pt"

    with pytest.raises(ModelNotFoundError) as excinfo:
        load_model(missing)

    assert "best.pt" in str(excinfo.value)
    assert "train.py" in str(excinfo.value)


def test_detect_image_converts_ultralytics_result_to_supervision_detections():
    model = _FakeModel(
        [
            _FakeResult(
                (
                    np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32),
                    np.array([0.88], dtype=np.float32),
                    np.array([1], dtype=np.float32),
                ),
                {1: "夹杂"},
            )
        ]
    )
    image = np.zeros((120, 120, 3), dtype=np.uint8)

    detections = detect_image(model, image, conf_threshold=0.3, class_names=["龟裂", "夹杂"])

    assert isinstance(detections, sv.Detections)
    assert len(detections) == 1
    assert detections.class_id.tolist() == [1]
    assert detections.confidence.tolist() == pytest.approx([0.88])
    assert model.calls[0]["conf"] == 0.3


def test_detect_image_returns_empty_detections_when_model_finds_nothing():
    model = _FakeModel(
        [
            _FakeResult(
                (
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                ),
                {},
            )
        ]
    )
    image = np.zeros((120, 120, 3), dtype=np.uint8)

    detections = detect_image(model, image, conf_threshold=0.25, class_names=[])

    assert len(detections) == 0


def test_detect_image_raises_detection_error_when_inference_fails():
    class _BrokenModel:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("boom")

    image = np.zeros((120, 120, 3), dtype=np.uint8)

    with pytest.raises(DetectionError):
        detect_image(_BrokenModel(), image, conf_threshold=0.25, class_names=[])


def test_detect_image_rejects_empty_image():
    model = _FakeModel([])

    with pytest.raises(ValueError):
        detect_image(model, np.zeros((0, 0, 3), dtype=np.uint8), conf_threshold=0.25, class_names=[])
