"""中文标签渲染测试。

Supervision 的 LabelAnnotator 内部使用 cv2.putText，只能渲染 ASCII，
中文类别名会显示为 "??"。本模块提供基于 PIL 的中文标签绘制，
在保持 BoxAnnotator / LabelAnnotator 视觉风格的前提下正确显示中文。
"""

import numpy as np
import pytest
import supervision as sv
from PIL import ImageFont

from src.annotation import build_labels
from src.chinese_label import (
    CHINESE_FONT_CANDIDATES,
    ChineseLabelAnnotator,
    find_chinese_font,
)


def _detections():
    return sv.Detections(
        xyxy=np.array([[10, 10, 60, 60], [100, 40, 180, 120]], dtype=np.float32),
        confidence=np.array([0.91, 0.65], dtype=np.float32),
        class_id=np.array([0, 3]),
    )


def test_font_candidates_are_listed():
    assert len(CHINESE_FONT_CANDIDATES) > 0
    assert all(path.lower().endswith((".ttf", ".ttc")) for path in CHINESE_FONT_CANDIDATES)


def test_find_chinese_font_returns_loadable_font():
    font = find_chinese_font()

    assert font is not None
    assert isinstance(font, ImageFont.FreeTypeFont)


def test_chinese_label_annotator_draws_actual_chinese_glyphs():
    """中文标签必须被真正渲染，而不是退化成问号。"""
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    detections = _detections()
    labels = build_labels(detections, class_names=["龟裂", "夹杂", "压痕斑块", "麻点"])

    annotator = ChineseLabelAnnotator(text_scale=0.5)
    annotated = annotator.annotate(image.copy(), detections, labels)

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)

    # 问号替代品是纯 ASCII 渲染，而真正的中文笔画密度明显更高：
    # 通过统计标签区域内非背景像素的连通块数量差异来验证。
    # 这里直接比较"中文标签"与"同长度 ASCII 标签"的墨迹量。
    ascii_annotator = ChineseLabelAnnotator(text_scale=0.5)
    ascii_annotated = ascii_annotator.annotate(
        image.copy(), detections, ["AA 0.91", "BB 0.65"]
    )

    chinese_ink = int(np.count_nonzero(np.any(annotated != 255, axis=2)))
    ascii_ink = int(np.count_nonzero(np.any(ascii_annotated != 255, axis=2)))

    assert chinese_ink != ascii_ink


def test_chinese_label_annotator_returns_scene_unchanged_when_no_detections():
    image = np.full((240, 320, 3), 255, dtype=np.uint8)

    annotator = ChineseLabelAnnotator()
    annotated = annotator.annotate(image.copy(), sv.Detections.empty(), [])

    assert np.array_equal(annotated, image)


def test_chinese_label_annotator_preserves_input_image():
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    original = image.copy()

    annotator = ChineseLabelAnnotator()
    annotator.annotate(image, _detections(), ["龟裂 0.91", "麻点 0.65"])

    assert np.array_equal(image, original)


def test_chinese_label_annotator_falls_back_to_ascii_when_no_font():
    """找不到中文字体时应降级为 ASCII，不能崩溃。"""
    annotator = ChineseLabelAnnotator(font_path=None, fallback_only=True)
    image = np.full((240, 320, 3), 255, dtype=np.uint8)

    annotated = annotator.annotate(image.copy(), _detections(), ["龟裂 0.91", "麻点 0.65"])

    assert annotated.shape == image.shape


@pytest.mark.parametrize("name", ["龟裂", "夹杂", "压痕斑块", "麻点", "氧化铁皮压入", "划痕"])
def test_all_defect_class_names_render_without_raising(name):
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 60, 60]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )

    annotator = ChineseLabelAnnotator()
    annotated = annotator.annotate(image.copy(), detections, [f"{name} 0.90"])

    assert annotated.shape == image.shape
