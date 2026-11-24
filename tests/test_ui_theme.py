"""UI 主题模块测试。

验证设计 token、置信度语义分级、九宫格热力 tile 生成的正确性。
视觉呈现本身需人工核验，这里锁定可测的逻辑部分。
"""

import re

import supervision as sv
import numpy as np
import pytest

from src.ui_theme import (
    SEVERITY_COLORS,
    THEME_CSS,
    build_zone_tiles_html,
    severity_color,
    severity_level,
)
from src.zone_analysis import analyze_zones


def test_theme_css_defines_core_design_tokens():
    """主题必须定义完整的设计 token，避免组件各写各的色值。"""
    for token in [
        "--brand",
        "--brand-strong",
        "--brand-soft",
        "--surface",
        "--bg",
        "--border",
        "--text",
        "--text-2",
        "--success",
        "--warning",
        "--danger",
        "--info",
        "--radius-card",
        "--space-4",
    ]:
        assert token in THEME_CSS, f"缺少设计 token: {token}"


def test_theme_css_keeps_focus_visible_for_accessibility():
    """无障碍：键盘焦点必须有可见轮廓（UX-005）。"""
    assert ":focus-visible" in THEME_CSS
    assert "outline" in THEME_CSS


def test_theme_css_does_not_shrink_body_text_below_legible_size():
    """正文不得小于 14px（桌面），避免可读性下降。"""
    font_sizes = [int(size) for size in re.findall(r"font-size:\s*(\d+)px", THEME_CSS)]
    assert font_sizes, "主题中应有显式字号定义"
    assert min(font_sizes) >= 11, f"存在过小字号: {min(font_sizes)}px"


@pytest.mark.parametrize(
    "confidence,expected",
    [
        (1.0, "high"),
        (0.70, "high"),
        (0.699, "medium"),
        (0.40, "medium"),
        (0.399, "low"),
        (0.0, "low"),
    ],
)
def test_severity_level_boundaries(confidence, expected):
    """置信度分级边界：>=0.70 高、>=0.40 中、其余低。"""
    assert severity_level(confidence) == expected


def test_severity_colors_are_distinct_and_defined():
    """三档严重度必须有各自独立的色值。"""
    assert set(SEVERITY_COLORS) == {"high", "medium", "low"}
    assert len(set(SEVERITY_COLORS.values())) == 3


def test_severity_color_matches_level():
    assert severity_color(0.9) == SEVERITY_COLORS["high"]
    assert severity_color(0.5) == SEVERITY_COLORS["medium"]
    assert severity_color(0.1) == SEVERITY_COLORS["low"]


def _analysis_with_counts():
    detections = sv.Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],
                [20, 20, 60, 60],
                [120, 120, 180, 180],
            ],
            dtype=np.float32,
        ),
        confidence=np.array([0.9, 0.8, 0.5], dtype=np.float32),
        class_id=np.array([0, 0, 1]),
    )
    return analyze_zones(detections, width=300, height=300)


def test_build_zone_tiles_html_renders_nine_cells():
    html = build_zone_tiles_html(_analysis_with_counts())

    assert html.count('class="sv-zone-tile') >= 9


def test_build_zone_tiles_html_shows_zone_names_and_counts():
    analysis = _analysis_with_counts()
    html = build_zone_tiles_html(analysis)

    for name in ["左上", "中上", "右上", "左中", "中央", "右中", "左下", "中下", "右下"]:
        assert name in html
    # 左上 2 个缺陷
    assert ">2<" in html


def test_build_zone_tiles_html_marks_hotspot():
    analysis = _analysis_with_counts()
    html = build_zone_tiles_html(analysis)

    assert analysis.hotspot == "左上"
    assert "is-hotspot" in html


def test_build_zone_tiles_html_handles_zero_defects_gracefully():
    """空结果也应渲染完整九宫格，不能是空白（UX 反模式：no empty states）。"""
    analysis = analyze_zones(sv.Detections.empty(), width=300, height=300)

    html = build_zone_tiles_html(analysis)

    assert html.count('class="sv-zone-tile') >= 9
    assert "is-hotspot" not in html


def test_build_zone_tiles_html_includes_legend():
    """图例说明冷热含义，不只靠颜色传达信息（UX-003）。"""
    html = build_zone_tiles_html(_analysis_with_counts())

    assert "少" in html and "多" in html
