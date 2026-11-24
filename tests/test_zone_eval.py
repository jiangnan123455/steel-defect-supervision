"""区域分析人工核验测试。

方案书 8.3 节要求：区域分析指标需使用人工核验的测试标注计算，
不应只用模型自身输出作为唯一真值。

NEU-DET 测试集的 YOLO 标签即人工真值。本模块把"真值区域分布"与
"模型预测区域分布"逐图比对，得到方案书要求的两个指标：
- 区域计数绝对误差
- 热点区域判断准确率
"""

import numpy as np
import pytest
import supervision as sv

from src.zone_eval import (
    ZoneComparison,
    yolo_label_to_detections,
    compare_zone_distributions,
)


def test_yolo_label_to_detections_converts_normalized_xywh():
    """YOLO 标签 (class_id, cx, cy, w, h) 归一化坐标应转换为像素 xyxy。"""
    labels = [
        (0, 0.25, 0.25, 0.5, 0.5),
        (5, 0.75, 0.75, 0.5, 0.5),
    ]

    detections = yolo_label_to_detections(labels, width=200, height=100)

    assert len(detections) == 2
    np.testing.assert_allclose(detections.xyxy[0], [0, 0, 100, 50], atol=1e-6)
    np.testing.assert_allclose(detections.xyxy[1], [100, 50, 200, 100], atol=1e-6)
    assert detections.class_id.tolist() == [0, 5]


def test_yolo_label_to_detections_handles_empty_label():
    detections = yolo_label_to_detections([], width=200, height=100)

    assert len(detections) == 0


def test_compare_zone_distributions_reports_per_zone_error():
    """真值与预测在同一图上的区域计数差值应逐区域累加。"""
    ground_truth = sv.Detections(
        xyxy=np.array([[10, 10, 60, 60], [120, 120, 180, 180]], dtype=np.float32),
        confidence=np.array([1.0, 1.0], dtype=np.float32),
        class_id=np.array([0, 1]),
    )
    predictions = sv.Detections(
        xyxy=np.array([[10, 10, 60, 60]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )

    comparison = compare_zone_distributions(
        ground_truths=[ground_truth],
        predictions=[predictions],
        width=300,
        height=300,
        zone_names=("左上", "中上", "右上", "左中", "中央", "右中", "左下", "中下", "右下"),
    )

    # 真值：左上 1 + 中央 1；预测：左上 1 → 中央差 1，其余 0
    assert comparison.total_error == 1
    assert comparison.zone_errors["中央"] == 1
    assert comparison.zone_errors["左上"] == 0
    assert comparison.mean_error_per_image == pytest.approx(1.0)


def test_compare_zone_distributions_accumulates_across_images():
    first_gt = sv.Detections(
        xyxy=np.array([[10, 10, 60, 60]], dtype=np.float32),
        class_id=np.array([0]),
    )
    first_pred = sv.Detections(
        xyxy=np.array([[200, 200, 280, 280]], dtype=np.float32),
        class_id=np.array([0]),
    )
    second_gt = sv.Detections(
        xyxy=np.array([[120, 120, 180, 180]], dtype=np.float32),
        class_id=np.array([1]),
    )
    second_pred = sv.Detections(
        xyxy=np.array([[120, 120, 180, 180]], dtype=np.float32),
        class_id=np.array([1]),
    )

    comparison = compare_zone_distributions(
        ground_truths=[first_gt, second_gt],
        predictions=[first_pred, second_pred],
        width=300,
        height=300,
        zone_names=("左上", "中上", "右上", "左中", "中央", "右中", "左下", "中下", "右下"),
    )

    # 图1：真值左上1 vs 预测右下1 → 差2；图2：完全一致 → 差0
    assert comparison.total_error == 2
    assert comparison.per_image_errors == [2, 0]


def test_hotspot_agreement_counts_matching_images():
    """真值热点区域与预测热点区域一致的图片数即为命中数。"""
    gt_hotspot_center = sv.Detections(
        xyxy=np.array([[120, 120, 180, 180]], dtype=np.float32),
        class_id=np.array([0]),
    )
    pred_hotspot_center = gt_hotspot_center
    pred_hotspot_other = sv.Detections(
        xyxy=np.array([[10, 10, 60, 60]], dtype=np.float32),
        class_id=np.array([0]),
    )
    gt_empty = sv.Detections.empty()
    pred_empty = sv.Detections.empty()

    comparison = compare_zone_distributions(
        ground_truths=[gt_hotspot_center, gt_hotspot_center, gt_empty],
        predictions=[pred_hotspot_center, pred_hotspot_other, pred_empty],
        width=300,
        height=300,
        zone_names=("左上", "中上", "右上", "左中", "中央", "右中", "左下", "中下", "右下"),
    )

    # 图1 一致（中央）；图2 不一致；图3 双方都无缺陷，也算一致
    assert comparison.hotspot_agreement == 2
    assert comparison.hotspot_total == 3
    assert comparison.hotspot_accuracy == pytest.approx(2 / 3)


def test_zone_comparison_dataclass_defaults():
    comparison = ZoneComparison(
        zone_errors={"左上": 0},
        total_error=0,
        per_image_errors=[],
        hotspot_agreement=0,
        hotspot_total=0,
        ground_truth_total=0,
        prediction_total=0,
    )

    assert comparison.hotspot_accuracy == 0.0
    assert comparison.mean_error_per_image == 0.0
