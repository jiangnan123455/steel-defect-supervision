"""图表样式测试：锁定轴标签规范（UX-077 P0）。"""

import pandas as pd

from src.ui_charts import class_bar_chart, zone_bar_chart


def test_zone_chart_has_axis_titles_with_units():
    """UX-077：坐标轴必须带标签与单位。"""
    frame = pd.DataFrame({"区域": ["左上", "中央"], "缺陷数量": [1, 3]})

    figure = zone_bar_chart(frame)

    assert figure.layout.xaxis.title.text == "区域"
    assert "个" in figure.layout.yaxis.title.text


def test_zone_chart_uses_single_series_without_legend():
    """单系列不需要图例，减少视觉噪声。"""
    frame = pd.DataFrame({"区域": ["左上"], "缺陷数量": [2]})

    figure = zone_bar_chart(frame)

    assert figure.layout.showlegend is False
    assert len(figure.data) == 1


def test_class_chart_is_horizontal_with_count_unit():
    frame = pd.DataFrame({"缺陷类别": ["龟裂", "夹杂"], "数量": [2, 1]})

    figure = class_bar_chart(frame)

    assert figure.data[0].orientation == "h"
    assert "个" in figure.layout.xaxis.title.text


def test_charts_transparent_background_for_card_layout():
    """图表底色透明，跟随卡片背景，避免出现灰块。"""
    frame = pd.DataFrame({"区域": ["左上"], "缺陷数量": [1]})

    figure = zone_bar_chart(frame)

    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.plot_bgcolor == "rgba(0,0,0,0)"
