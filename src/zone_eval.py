"""区域分析人工核验。

方案书 8.3 节要求：区域分析指标需使用人工核验的测试标注计算，
不应只用模型自身输出作为唯一真值。

NEU-DET 测试集的 YOLO 标签即人工真值。本模块把"真值区域分布"与
"模型预测区域分布"逐图比对，得到方案书要求的两个指标：
- 区域计数绝对误差（逐区域误差、总误差、单图平均误差）
- 热点区域判断准确率（真值与预测的缺陷集中区域一致的比例）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from src.zone_analysis import analyze_zones


@dataclass
class ZoneComparison:
    """真值与模型预测的区域分布比对结果。"""

    zone_errors: dict[str, int] = field(default_factory=dict)
    total_error: int = 0
    per_image_errors: list[int] = field(default_factory=list)
    hotspot_agreement: int = 0
    hotspot_total: int = 0
    ground_truth_total: int = 0
    prediction_total: int = 0

    @property
    def hotspot_accuracy(self) -> float:
        if self.hotspot_total == 0:
            return 0.0
        return self.hotspot_agreement / self.hotspot_total

    @property
    def mean_error_per_image(self) -> float:
        if not self.per_image_errors:
            return 0.0
        return float(np.mean(self.per_image_errors))


def yolo_label_to_detections(
    labels: list[tuple[int, float, float, float, float]],
    width: int,
    height: int,
) -> sv.Detections:
    """把 YOLO 归一化标签转换为 Supervision 检测结果（人工真值）。

    Args:
        labels: 每项为 `(class_id, cx, cy, w, h)`，坐标按宽高归一化到 [0, 1]。
        width: 图像宽度（像素）。
        height: 图像高度（像素）。

    Returns:
        对应的 `sv.Detections`，置信度固定为 1.0（真值没有置信度概念）。
    """
    if not labels:
        return sv.Detections.empty()

    xyxy = []
    class_ids = []
    for class_id, cx, cy, w, h in labels:
        x1 = (cx - w / 2) * width
        y1 = (cy - h / 2) * height
        x2 = (cx + w / 2) * width
        y2 = (cy + h / 2) * height
        xyxy.append([x1, y1, x2, y2])
        class_ids.append(class_id)

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.ones(len(xyxy), dtype=np.float32),
        class_id=np.array(class_ids, dtype=np.int64),
    )


def compare_zone_distributions(
    ground_truths: list[sv.Detections],
    predictions: list[sv.Detections],
    width: int,
    height: int,
    zone_names: tuple[str, ...],
) -> ZoneComparison:
    """逐图比对真值与预测的区域分布。

    Args:
        ground_truths: 每张图的人工真值检测结果。
        predictions: 每张图的模型预测结果，与真值一一对应。
        width: 图像宽度（像素）。
        height: 图像高度（像素）。
        zone_names: 区域名称，按行优先顺序。

    Returns:
        `ZoneComparison`，包含逐区域误差、总误差、热点命中率等。
    """
    if len(ground_truths) != len(predictions):
        raise ValueError(
            f"真值与预测数量不一致：{len(ground_truths)} vs {len(predictions)}"
        )

    zone_errors = {name: 0 for name in zone_names}
    per_image_errors: list[int] = []
    hotspot_agreement = 0
    hotspot_total = 0
    ground_truth_total = 0
    prediction_total = 0

    for ground_truth, prediction in zip(ground_truths, predictions):
        truth_analysis = analyze_zones(ground_truth, width=width, height=height)
        prediction_analysis = analyze_zones(prediction, width=width, height=height)

        ground_truth_total += truth_analysis.total
        prediction_total += prediction_analysis.total

        image_error = 0
        for name in zone_names:
            diff = abs(truth_analysis.counts.get(name, 0) - prediction_analysis.counts.get(name, 0))
            zone_errors[name] += diff
            image_error += diff
        per_image_errors.append(image_error)
        total_error = sum(per_image_errors)

        truth_hotspot = truth_analysis.hotspot
        prediction_hotspot = prediction_analysis.hotspot
        hotspot_total += 1
        if truth_hotspot == prediction_hotspot:
            hotspot_agreement += 1

    return ZoneComparison(
        zone_errors=zone_errors,
        total_error=total_error,
        per_image_errors=per_image_errors,
        hotspot_agreement=hotspot_agreement,
        hotspot_total=hotspot_total,
        ground_truth_total=ground_truth_total,
        prediction_total=prediction_total,
    )
