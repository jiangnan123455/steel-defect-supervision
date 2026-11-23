"""YOLO 缺陷检测推理封装。

模型负责识别缺陷类别、置信度和边界框；本模块只负责加载模型、执行推理，
以及把 Ultralytics 结果转换为统一的 Supervision `sv.Detections`。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv

# NEU-DET 钢材表面缺陷的六个类别（顺序对应训练时的 class_id）
DEFAULT_CLASS_NAMES: list[str] = [
    "龟裂",
    "夹杂",
    "压痕斑块",
    "麻点",
    "氧化铁皮压入",
    "划痕",
]

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "best.pt"


class DetectionError(RuntimeError):
    """检测过程发生异常。"""


class ModelNotFoundError(DetectionError):
    """模型权重文件缺失。"""


def load_model(weights_path: str | Path | None = None, device: str | None = None):
    """加载 YOLO 权重文件。

    Args:
        weights_path: 权重文件路径，默认为 `weights/best.pt`。
        device: 推理设备，例如 `"cpu"` 或 `"0"`。为 None 时由 Ultralytics 自动选择。

    Returns:
        已加载的 Ultralytics YOLO 模型。

    Raises:
        ModelNotFoundError: 权重文件不存在时抛出，提示如何获取模型。
    """
    path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
    if not path.exists():
        raise ModelNotFoundError(
            f"未找到模型权重文件：{path}\n"
            "请先运行 train.py 训练缺陷检测模型，或把已有的 best.pt 放入 weights/ 目录。"
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - 依赖缺失时的兜底提示
        raise DetectionError(
            "未安装 ultralytics，请先执行：pip install ultralytics"
        ) from error

    try:
        model = YOLO(str(path))
        if device:
            model.to(device)
    except Exception as error:
        raise DetectionError(f"加载模型失败：{error}") from error

    return model


def detect_image(
    model,
    image: np.ndarray,
    conf_threshold: float = 0.25,
    class_names: list[str] | None = None,
    iou: float = 0.7,
) -> sv.Detections:
    """对单张图片执行检测，返回 Supervision 检测结果。

    Args:
        model: 已加载的 YOLO 模型。
        image: BGR 格式图片。
        conf_threshold: 置信度阈值。
        class_names: 类别名称列表，用于在模型未自带名称时补充。
        iou: NMS 的 IoU 阈值。

    Returns:
        `sv.Detections` 检测结果。

    Raises:
        ValueError: 图片为空。
        DetectionError: 推理过程异常。
    """
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("输入图片为空，无法执行检测。")

    try:
        results = model(image, conf=conf_threshold, iou=iou, verbose=False)
    except Exception as error:
        raise DetectionError(f"模型推理失败：{error}") from error

    if results is None or len(results) == 0:
        return sv.Detections.empty()

    result = results[0]

    try:
        detections = sv.Detections.from_ultralytics(result)
    except Exception as error:
        raise DetectionError(f"检测结果转换失败：{error}") from error

    if len(detections) and class_names:
        detections = _apply_class_names(detections, class_names)

    return detections


def _apply_class_names(detections: sv.Detections, class_names: list[str]) -> sv.Detections:
    """用外部类别名覆盖模型自带的名称，便于展示中文缺陷类别。"""
    from supervision.detection.core import CLASS_NAME_DATA_FIELD

    class_id = detections.class_id
    if class_id is None:
        return detections

    names = []
    for value in class_id:
        index = int(value)
        names.append(class_names[index] if 0 <= index < len(class_names) else f"class_{index}")

    detections.data[CLASS_NAME_DATA_FIELD] = np.array(names, dtype=str)
    return detections
