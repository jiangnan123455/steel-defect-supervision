"""Supervision 标注逻辑测试。"""

import numpy as np
import supervision as sv

from src.annotation import annotate_image, build_labels


def _detections():
    return sv.Detections(
        xyxy=np.array([[10, 10, 60, 60], [100, 40, 180, 120]], dtype=np.float32),
        confidence=np.array([0.91, 0.65], dtype=np.float32),
        class_id=np.array([0, 3]),
    )


def test_build_labels_contains_class_name_and_confidence():
    labels = build_labels(_detections(), class_names=["龟裂", "夹杂", "压痕斑块", "麻点"])

    assert labels == ["龟裂 0.91", "麻点 0.65"]


def test_build_labels_falls_back_to_class_id_when_name_missing():
    labels = build_labels(_detections(), class_names=["龟裂"])

    assert labels == ["龟裂 0.91", "class_3 0.65"]


def test_build_labels_handles_empty_detections():
    assert build_labels(sv.Detections.empty(), class_names=["龟裂"]) == []


def test_annotate_image_returns_same_shape_as_input():
    image = np.full((240, 320, 3), 255, dtype=np.uint8)

    annotated = annotate_image(image, _detections(), class_names=["龟裂", "夹杂", "压痕斑块", "麻点"])

    assert annotated.shape == image.shape
    assert annotated.dtype == np.uint8


def test_annotate_image_does_not_mutate_input():
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    original = image.copy()

    annotate_image(image, _detections(), class_names=["龟裂", "夹杂", "压痕斑块", "麻点"])

    assert np.array_equal(image, original)


def test_annotate_image_draws_ink_on_the_scene():
    image = np.full((240, 320, 3), 255, dtype=np.uint8)

    annotated = annotate_image(image, _detections(), class_names=["龟裂", "夹杂", "压痕斑块", "麻点"])

    assert not np.array_equal(annotated, image)
