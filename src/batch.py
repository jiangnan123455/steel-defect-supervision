"""批量检测：逐张推理、汇总统计、单张失败不影响整批。

与单张检测共用同一套推理、标注与统计逻辑，区别只在于调度：
一次接收多张图片，逐张调用 `detect_image`，把每张的结果整理成可展示、可导出的记录。

设计约束（对应 `docs/batch_mode.md`）：
- 逐张推理，不是并行；单批数量与单张体积都有上限。
- 某张图片读取或检测失败时记录原因，继续处理后面的图片。
- 汇总口径：已完成张数、失败张数、累计缺陷数、按类别的合计。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import supervision as sv

from src.annotation import annotate_image
from src.detection import DEFAULT_CLASS_NAMES, detect_image
from src.utils import decode_image_bytes, summarize_detections, validate_upload

# 单批最多处理的图片数：超过的部分直接拒绝，避免一次上传把内存吃满
MAX_BATCH_FILES = 20

STATUS_DONE = "已完成"
STATUS_EMPTY = "已完成 · 未检出"
STATUS_FAILED = "失败"


@dataclass
class BatchItem:
    """一张图片在批次里的结果。"""

    name: str
    status: str
    width: int = 0
    height: int = 0
    count: int = 0
    elapsed_ms: float = 0.0
    error: str = ""
    annotated: np.ndarray | None = None
    detections: sv.Detections | None = None
    class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != STATUS_FAILED


@dataclass
class BatchResult:
    """整批结果：逐张记录 + 汇总口径。"""

    items: list[BatchItem]
    conf_threshold: float
    elapsed_ms: float

    @property
    def completed(self) -> int:
        return sum(1 for item in self.items if item.ok)

    @property
    def failed(self) -> int:
        return sum(1 for item in self.items if not item.ok)

    @property
    def total_defects(self) -> int:
        return sum(item.count for item in self.items if item.ok)

    @property
    def class_counts(self) -> dict[str, int]:
        """整批按类别合计，按数量降序。"""
        merged: dict[str, int] = {}
        for item in self.items:
            for name, count in item.class_counts.items():
                merged[name] = merged.get(name, 0) + count
        return dict(sorted(merged.items(), key=lambda pair: pair[1], reverse=True))

    @property
    def elapsed_text(self) -> str:
        return f"{self.elapsed_ms / 1000:.1f} s"


def run_batch(
    model,
    files: list[tuple[str, bytes]],
    conf_threshold: float,
    class_names: list[str] | None = None,
    detect_fn=None,
    progress=None,
) -> BatchResult:
    """逐张检测一批图片。

    Args:
        model: 已加载的模型；`detect_fn` 已注入时可传 None。
        files: `(文件名, 图片字节)` 列表，顺序即处理顺序。
        conf_threshold: 本批次统一的置信度阈值。
        class_names: 类别名列表，缺省用 `DEFAULT_CLASS_NAMES`。
        detect_fn: 推理函数，签名 `(model, image, conf_threshold, class_names)`；
            默认 `detect_image`，测试时可注入桩函数。
        progress: 进度回调 `(已完成张数, 总张数, 当前文件名)`，用于页面进度条。

    Returns:
        `BatchResult`。单张失败只记录在对应 `BatchItem` 上，不抛出。
    """
    names = class_names or DEFAULT_CLASS_NAMES
    detector = detect_fn or detect_image
    # 超出上限的部分不处理，交由调用方提示，避免静默截断
    selected = files[:MAX_BATCH_FILES]
    total = len(selected)

    items: list[BatchItem] = []
    started = time.perf_counter()

    for index, (filename, raw) in enumerate(selected):
        if progress is not None:
            progress(index, total, filename)

        item = BatchItem(name=filename, status=STATUS_DONE)
        try:
            validate_upload(filename, len(raw))
            image = decode_image_bytes(raw)
            item.height, item.width = image.shape[:2]

            single_start = time.perf_counter()
            detections = detector(model, image, conf_threshold, names)
            item.elapsed_ms = (time.perf_counter() - single_start) * 1000

            summary = summarize_detections(detections, names)
            item.detections = detections
            item.count = int(summary["total"])
            item.class_counts = dict(summary["class_counts"])
            item.annotated = annotate_image(
                image,
                detections,
                names,
                text_scale=0.34 if min(item.height, item.width) < 320 else 0.5,
                with_confidence=min(item.height, item.width) >= 320,
            )
            item.status = STATUS_EMPTY if item.count == 0 else STATUS_DONE
        except Exception as exc:  # noqa: BLE001 - 单张失败必须不影响整批
            item.status = STATUS_FAILED
            item.error = str(exc) or exc.__class__.__name__

        items.append(item)

    if progress is not None:
        progress(total, total, "")

    return BatchResult(
        items=items,
        conf_threshold=conf_threshold,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def batch_rows(result: BatchResult) -> list[dict]:
    """把整批结果摊平成明细表：一个检出占一行，未检出/失败的图片也各占一行。"""
    rows: list[dict] = []
    for item in result.items:
        if not item.ok:
            rows.append(
                {
                    "图片名称": item.name,
                    "缺陷类型": "—",
                    "置信度": None,
                    "检测状态": f"{STATUS_FAILED}：{item.error}",
                }
            )
            continue

        if item.count == 0 or item.detections is None or item.detections.confidence is None:
            rows.append(
                {
                    "图片名称": item.name,
                    "缺陷类型": "—",
                    "置信度": None,
                    "检测状态": STATUS_EMPTY,
                }
            )
            continue

        for confidence, class_id in zip(item.detections.confidence, item.detections.class_id):
            rows.append(
                {
                    "图片名称": item.name,
                    "缺陷类型": DEFAULT_CLASS_NAMES[int(class_id)],
                    "置信度": round(float(confidence), 3),
                    "检测状态": STATUS_DONE,
                }
            )
    return rows


def batch_frame(result: BatchResult) -> pd.DataFrame:
    """明细表用的 DataFrame（列顺序固定，便于导出与展示一致）。"""
    return pd.DataFrame(
        batch_rows(result), columns=["图片名称", "缺陷类型", "置信度", "检测状态"]
    )


def batch_csv(result: BatchResult) -> bytes:
    """批量明细 CSV（UTF-8 BOM，Excel 直接打开不乱码）。"""
    return batch_frame(result).to_csv(index=False).encode("utf-8-sig")


def item_captions(result: BatchResult) -> list[tuple[str, str, bool]]:
    """画布缩略图用的 `(文件名, 说明文字, 是否失败)` 列表。"""
    captions: list[tuple[str, str, bool]] = []
    for item in result.items:
        if not item.ok:
            captions.append((item.name, f"失败 · {item.error}", True))
        elif item.count == 0:
            captions.append((item.name, "未检出", False))
        else:
            captions.append((item.name, f"{item.count} 个缺陷", False))
    return captions
