"""Web 页面测试（对应项目方案书第 11 章功能测试）。"""

from pathlib import Path

import numpy as np
import pytest
import supervision as sv
from streamlit.testing.v1 import AppTest

from src.utils import decode_image_bytes, encode_png_bytes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app.py"


@pytest.fixture()
def app_test():
    return AppTest.from_file(str(APP_PATH), default_timeout=120)


def test_page_renders_without_exception(app_test):
    app_test.run()

    assert len(app_test.exception) == 0


def test_page_shows_title_and_project_caption(app_test):
    app_test.run()

    titles = [item.value for item in app_test.title]
    assert any("钢材表面缺陷" in title for title in titles)


def test_page_reports_model_missing_without_crashing(app_test, monkeypatch):
    """T04：模型权重缺失时给出提示，程序不崩溃。"""
    monkeypatch.setenv(
        "STEEL_WEIGHTS_PATH", str(PROJECT_ROOT / "weights" / "nonexistent.pt")
    )

    app_test.run()

    assert len(app_test.exception) == 0
    errors = [item.value for item in app_test.error]
    assert any("模型" in message for message in errors)


def test_page_reports_model_ready_when_weights_exist(app_test, monkeypatch):
    """权重就位时，侧栏显示模型已就绪。"""
    monkeypatch.setenv(
        "STEEL_WEIGHTS_PATH", str(PROJECT_ROOT / "weights" / "best.pt")
    )

    app_test.run()

    assert len(app_test.exception) == 0
    successes = [item.value for item in app_test.success]
    assert any("就绪" in message for message in successes)


def test_page_shows_confidence_slider(app_test):
    app_test.run()

    sliders = app_test.slider
    assert any(slider.label == "置信度阈值" for slider in sliders)


def test_page_offers_example_images_when_available(app_test):
    app_test.run()

    expander_titles = [item.label for item in app_test.expander]
    assert any("示例图片" in (title or "") for title in expander_titles)


def test_invalid_image_payload_is_rejected():
    """T03：无效图片数据给出明确错误，而不是崩溃。"""
    with pytest.raises(ValueError):
        decode_image_bytes(b"definitely-not-an-image")


def test_png_roundtrip_preserves_dimensions():
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    image[10:20, 30:40] = 255

    decoded = decode_image_bytes(encode_png_bytes(image))

    assert decoded.shape == (60, 80, 3)


def test_zone_counts_sum_equals_detection_count_for_real_detection():
    """T08 的核心性质：区域计数之和等于参与统计的检测目标数。"""
    from src.zone_analysis import analyze_zones

    detections = sv.Detections(
        xyxy=np.array(
            [[10, 10, 60, 60], [120, 20, 200, 90], [30, 200, 100, 280]],
            dtype=np.float32,
        ),
        confidence=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        class_id=np.array([0, 1, 2]),
    )

    analysis = analyze_zones(detections, width=300, height=300)

    assert sum(analysis.counts.values()) == analysis.assigned_count == 3
