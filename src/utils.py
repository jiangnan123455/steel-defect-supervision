"""图片处理与统计工具。"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
import supervision as sv

SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def validate_upload(filename: str, size_bytes: int) -> None:
    """校验上传文件是否可用。

    Args:
        filename: 文件名。
        size_bytes: 文件字节数。

    Raises:
        ValueError: 格式不支持、文件为空或超过大小上限。
    """
    extension = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"不支持的文件格式：{extension or '未知'}。仅支持 JPG、JPEG、PNG。"
        )
    if size_bytes <= 0:
        raise ValueError("上传的文件为空，请重新选择。")
    if size_bytes > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"图片体积超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB，请压缩后重试。"
        )


def decode_image_bytes(data: bytes) -> np.ndarray:
    """把图片字节解码为 BGR 数组。

    Args:
        data: 图片原始字节。

    Returns:
        BGR 格式的 numpy 数组。

    Raises:
        ValueError: 数据为空或无法解码。
    """
    if not data:
        raise ValueError("图片数据为空，无法读取。")

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解析该图片，请确认文件没有损坏且格式为 JPG 或 PNG。")
    return image


def encode_png_bytes(image: np.ndarray) -> bytes:
    """把图像编码为 PNG 字节，便于在 Web 页面下载。"""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("图片编码失败。")
    return buffer.tobytes()


def summarize_detections(
    detections: sv.Detections,
    class_names: list[str] | None = None,
) -> dict:
    """汇总检测结果，返回总数、类别分布与平均置信度。

    Args:
        detections: 检测结果。
        class_names: 类别名称列表。

    Returns:
        包含 `total`、`class_counts`、`mean_confidence` 的字典。
    """
    names = class_names or []
    total = len(detections)

    class_counts: Counter[str] = Counter()
    if total and detections.class_id is not None:
        for value in detections.class_id:
            index = int(value)
            name = names[index] if 0 <= index < len(names) else f"class_{index}"
            class_counts[name] += 1

    mean_confidence = 0.0
    if total and detections.confidence is not None:
        mean_confidence = float(np.mean(detections.confidence))

    return {
        "total": total,
        "class_counts": dict(class_counts.most_common()),
        "mean_confidence": mean_confidence,
    }


def detections_to_records(
    detections: sv.Detections,
    class_names: list[str] | None = None,
) -> list[dict]:
    """把检测结果转成表格用的字典列表。

    Args:
        detections: 检测结果。
        class_names: 类别名称列表。

    Returns:
        每行包含 `序号`、`类别`、`置信度`、`x1`、`y1`、`x2`、`y2` 的字典列表。
    """
    names = class_names or []
    records: list[dict] = []

    for index in range(len(detections)):
        x1, y1, x2, y2 = (float(value) for value in detections.xyxy[index])

        if detections.class_id is None:
            class_name = "unknown"
        else:
            class_value = int(detections.class_id[index])
            class_name = names[class_value] if 0 <= class_value < len(names) else f"class_{class_value}"

        confidence = (
            float(detections.confidence[index]) if detections.confidence is not None else float("nan")
        )

        records.append(
            {
                "序号": index + 1,
                "类别": class_name,
                "置信度": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "宽度": x2 - x1,
                "高度": y2 - y1,
            }
        )

    return records
