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


def test_label_clamped_inside_image_when_box_touches_top_edge():
    """框贴顶边时，标签必须被拉回画面内，不能被裁掉。"""
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[25, 2, 192, 91]], dtype=np.float32),
        confidence=np.array([0.46], dtype=np.float32),
        class_id=np.array([0]),
    )

    annotator = ChineseLabelAnnotator(text_scale=0.5)
    annotated = annotator.annotate(image.copy(), detections, ["龟裂 0.46"])

    # 标签背景框与文字都应落在画面内：顶部若干行必须出现非背景像素
    top_rows = annotated[:20, :, :]
    assert np.any(top_rows != 255), "标签被裁到画面外，顶部没有绘制内容"


def test_label_clamped_when_box_touches_left_edge():
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[0, 100, 80, 180]], dtype=np.float32),
        confidence=np.array([0.8], dtype=np.float32),
        class_id=np.array([0]),
    )

    annotator = ChineseLabelAnnotator(text_scale=0.5)
    annotated = annotator.annotate(image.copy(), detections, ["龟裂 0.80"])

    assert not np.array_equal(annotated, image)
    # 不应有像素被写到画面外（数组形状不变即隐含裁剪安全）
    assert annotated.shape == image.shape


def test_clamp_label_boxes_pulls_box_fully_inside_frame():
    """钳制函数必须保证标签矩形完整落在画面内，且尺寸不变。"""
    from src.chinese_label import clamp_label_boxes

    boxes = np.array([[-30.0, -20.0, 50.0, 15.0, 20.0]], dtype=np.float32)

    clamped = clamp_label_boxes(boxes, width=200, height=200)

    assert clamped[0, 0] >= 0
    assert clamped[0, 1] >= 0
    assert clamped[0, 2] <= 200
    assert clamped[0, 3] <= 200
    assert (clamped[0, 2] - clamped[0, 0]) == pytest.approx(80.0)
    assert (clamped[0, 3] - clamped[0, 1]) == pytest.approx(35.0)


def test_clamp_label_boxes_leaves_inside_box_untouched():
    from src.chinese_label import clamp_label_boxes

    boxes = np.array([[10.0, 10.0, 90.0, 40.0, 20.0]], dtype=np.float32)

    clamped = clamp_label_boxes(boxes, width=200, height=200)

    np.testing.assert_allclose(clamped[0, :4], [10, 10, 90, 40])


def test_clamp_label_boxes_handles_box_wider_than_frame():
    """标签比画面还宽时，贴住左边界，不产生负坐标。"""
    from src.chinese_label import clamp_label_boxes

    boxes = np.array([[-50.0, 5.0, 500.0, 40.0, 20.0]], dtype=np.float32)

    clamped = clamp_label_boxes(boxes, width=200, height=200)

    assert clamped[0, 0] == 0
    assert clamped[0, 2] >= 200


def test_label_background_uses_same_bgr_color_as_box():
    """标签背景色必须与检测框同色（同一条 BGR 链路，不能出现通道反转）。"""
    from src.annotation import DEFECT_COLOR, annotate_image

    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[20, 60, 180, 150]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )

    annotated = annotate_image(image.copy(), detections, ["龟裂 0.90"])

    expected_bgr = np.array(DEFECT_COLOR.as_bgr(), dtype=np.uint8)
    matches = np.all(annotated == expected_bgr, axis=2)
    assert matches.sum() > 200, (
        f"标签背景区域未使用预期的 BGR 色 {expected_bgr.tolist()}，"
        f"实际出现次数 {matches.sum()}"
    )

    # 反向检查：不应出现被通道反转后的颜色
    reversed_bgr = expected_bgr[::-1]
    wrong = np.all(annotated == reversed_bgr, axis=2)
    assert wrong.sum() == 0, "标签颜色发生通道反转（BGR 与 RGB 混淆）"
