"""批量检测的单元测试（对应 docs/batch_mode.md 的批量模式约定）。"""

import numpy as np
import pytest
import supervision as sv

from src.batch import (
    MAX_BATCH_FILES,
    STATUS_DONE,
    STATUS_EMPTY,
    STATUS_FAILED,
    batch_csv,
    batch_frame,
    batch_rows,
    item_captions,
    run_batch,
)
from src.detection import DEFAULT_CLASS_NAMES
from src.utils import encode_png_bytes


def _image_bytes(value: int = 128, size: int = 96) -> bytes:
    """造一张可解码的灰度图，value 用来区分不同图片。"""
    image = np.full((size, size, 3), value, dtype=np.uint8)
    return encode_png_bytes(image)


def _one_detection(model, image, conf_threshold, class_names):
    """桩推理：固定返回 1 个「龟裂」框，置信度 0.8。"""
    return sv.Detections(
        xyxy=np.array([[6, 6, 40, 40]], dtype=np.float32),
        confidence=np.array([0.8], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )


def _no_detection(model, image, conf_threshold, class_names):
    return sv.Detections.empty()


def test_run_batch_counts_and_status():
    files = [("a.png", _image_bytes(10)), ("b.png", _image_bytes(20))]

    result = run_batch(None, files, 0.25, DEFAULT_CLASS_NAMES, detect_fn=_one_detection)

    assert [item.status for item in result.items] == [STATUS_DONE, STATUS_DONE]
    assert result.completed == 2 and result.failed == 0
    assert result.total_defects == 2
    assert result.class_counts == {"龟裂": 2}
    assert result.items[0].annotated is not None


def test_run_batch_marks_empty_when_nothing_detected():
    files = [("a.png", _image_bytes())]

    result = run_batch(None, files, 0.25, DEFAULT_CLASS_NAMES, detect_fn=_no_detection)

    assert result.items[0].status == STATUS_EMPTY
    assert result.items[0].count == 0
    assert result.total_defects == 0
    assert result.class_counts == {}


def test_run_batch_continues_after_broken_file():
    """一张坏图不能中断整批：失败记录原因，后面的图片照常处理。"""
    files = [
        ("good-1.png", _image_bytes(10)),
        ("broken.png", b"definitely-not-an-image"),
        ("good-2.png", _image_bytes(30)),
    ]

    result = run_batch(None, files, 0.25, DEFAULT_CLASS_NAMES, detect_fn=_one_detection)

    assert [item.name for item in result.items] == ["good-1.png", "broken.png", "good-2.png"]
    broken = result.items[1]
    assert broken.status == STATUS_FAILED
    assert broken.error, "失败必须带上原因"
    assert result.completed == 2
    assert result.failed == 1
    assert result.total_defects == 2


def test_run_batch_reports_progress_for_every_file_and_finish():
    seen: list[tuple[int, int]] = []
    files = [("a.png", _image_bytes(10)), ("b.png", _image_bytes(20))]

    run_batch(
        None,
        files,
        0.25,
        DEFAULT_CLASS_NAMES,
        detect_fn=_one_detection,
        progress=lambda done, total, name: seen.append((done, total)),
    )

    assert seen[0] == (0, 2)
    assert seen[-1] == (2, 2)


def test_run_batch_caps_batch_size():
    files = [(f"{i}.png", _image_bytes(i % 200)) for i in range(MAX_BATCH_FILES + 5)]

    result = run_batch(None, files, 0.25, DEFAULT_CLASS_NAMES, detect_fn=_no_detection)

    assert len(result.items) == MAX_BATCH_FILES


def test_batch_rows_expands_detections_and_keeps_failed_rows():
    files = [
        ("ok.png", _image_bytes(10)),
        ("bad.png", b"broken"),
        ("empty.png", _image_bytes(20)),
    ]
    calls = {"n": 0}

    def _first_only(model, image, conf_threshold, class_names):
        calls["n"] += 1
        if calls["n"] == 1:
            return _one_detection(model, image, conf_threshold, class_names)
        return sv.Detections.empty()

    result = run_batch(
        None,
        files,
        0.25,
        DEFAULT_CLASS_NAMES,
        detect_fn=_first_only,
    )

    rows = batch_rows(result)

    assert len(rows) == 3
    assert rows[0]["缺陷类型"] == "龟裂" and rows[0]["置信度"] == pytest.approx(0.8)
    assert rows[1]["检测状态"].startswith(STATUS_FAILED)
    assert rows[2]["检测状态"] == STATUS_EMPTY


def test_batch_frame_columns_are_fixed():
    result = run_batch(None, [("a.png", _image_bytes())], 0.25, detect_fn=_one_detection)

    frame = batch_frame(result)

    assert list(frame.columns) == ["图片名称", "缺陷类型", "置信度", "检测状态"]


def test_batch_csv_is_excel_friendly():
    result = run_batch(None, [("a.png", _image_bytes())], 0.25, detect_fn=_one_detection)

    payload = batch_csv(result)

    assert payload.startswith(b"\xef\xbb\xbf"), "CSV 需带 UTF-8 BOM，Excel 打开才不乱码"
    text = payload.decode("utf-8-sig")
    assert "图片名称" in text and "龟裂" in text


def test_item_captions_describe_failure_and_empty():
    files = [("ok.png", _image_bytes(10)), ("bad.png", b"broken"), ("none.png", _image_bytes(30))]
    result = run_batch(None, files, 0.25, detect_fn=_no_detection)

    captions = item_captions(result)

    assert captions[0][1] == "未检出"
    assert captions[1][2] is True and "失败" in captions[1][1]
    assert captions[2][1] == "未检出"
