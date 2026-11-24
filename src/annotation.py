"""Supervision 标注逻辑。

把 YOLO 的推理结果转换为统一的 `sv.Detections`，再使用 Supervision 的
`BoxAnnotator` 和 `LabelAnnotator` 绘制缺陷框与类别标签。
"""

from __future__ import annotations

import supervision as sv

from src.chinese_label import ChineseLabelAnnotator

DEFAULT_BOX_THICKNESS = 2
DEFAULT_LABEL_TEXT_SCALE = 0.5

# 缺陷统一使用告警红：与站内「红色 = 缺陷/告警」语义保持一致，
# 也避免 supervision 默认紫色调色板与品牌蓝产生视觉冲突。
DEFECT_COLOR = sv.Color(r=217, g=45, b=32)  # #D92D20


def build_labels(
    detections: sv.Detections,
    class_names: list[str] | None = None,
) -> list[str]:
    """为每个检测目标生成"类别名 + 置信度"的标签文本。

    Args:
        detections: 检测结果。
        class_names: 类别名称列表，下标对应 `class_id`。缺失时退化为
            `class_<id>` 形式。

    Returns:
        与检测目标一一对应的标签字符串列表。
    """
    names = class_names or []
    if len(detections) == 0:
        return []

    labels: list[str] = []
    class_id = detections.class_id
    confidence = detections.confidence

    for index in range(len(detections)):
        if class_id is None:
            name = "unknown"
        else:
            class_value = int(class_id[index])
            name = names[class_value] if 0 <= class_value < len(names) else f"class_{class_value}"

        if confidence is None:
            labels.append(name)
        else:
            labels.append(f"{name} {float(confidence[index]):.2f}")
    return labels


def annotate_image(
    image,
    detections: sv.Detections,
    class_names: list[str] | None = None,
    box_thickness: int = DEFAULT_BOX_THICKNESS,
    text_scale: float = DEFAULT_LABEL_TEXT_SCALE,
) -> "object":
    """在图像上绘制缺陷框与标签，返回标注后的新图像。

    原始图像不会被修改。

    Args:
        image: BGR 格式的原始图像（numpy 数组）。
        detections: 检测结果。
        class_names: 类别名称列表。
        box_thickness: 边框线宽。
        text_scale: 标签文字缩放比例。

    Returns:
        标注后的图像（numpy 数组）。
    """
    scene = image.copy()

    if len(detections) == 0:
        return scene

    labels = build_labels(detections, class_names)

    box_annotator = sv.BoxAnnotator(thickness=box_thickness, color=DEFECT_COLOR)
    # 使用中文标注器，确保"龟裂""夹杂"等缺陷类别名能正确显示，
    # 而不是 cv2.putText 渲染出的问号；标注器内部会把贴边标签拉回画面内。
    label_annotator = ChineseLabelAnnotator(text_scale=text_scale, color=DEFECT_COLOR)

    scene = box_annotator.annotate(scene=scene, detections=detections)
    scene = label_annotator.annotate(scene=scene, detections=detections, labels=labels)
    return scene
