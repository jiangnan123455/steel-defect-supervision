"""图表统一样式。

Plotly 图表集中在这里配置，保证配色、字体、轴标签规范一致。

依据 ui-ux-pro-max 指南：
- UX-077 [P0] 坐标轴必须带标签与单位
- UX-079 [P1] 图表配色需对色盲友好
- UX-080 [P2] 突出数据本身，不做装饰
- UX-039 [P1] 只保留一个强调色，克制使用
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.ui_theme import COLORS

_FONT_FAMILY = (
    'Inter, -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", '
    '"Noto Sans SC", sans-serif'
)

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor=COLORS["border"],
    gridwidth=1,
    zeroline=False,
    linecolor=COLORS["border_strong"],
    tickfont=dict(size=12, color=COLORS["text_2"], family=_FONT_FAMILY),
    title_font=dict(size=12, color=COLORS["text_2"], family=_FONT_FAMILY),
)


def _base_layout(height: int, x_title: str, y_title: str) -> dict:
    return dict(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT_FAMILY, size=12, color=COLORS["text_2"]),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border_strong"],
            font=dict(family=_FONT_FAMILY, size=12, color=COLORS["text"]),
        ),
        xaxis=dict(**_AXIS_STYLE, title=x_title),
        yaxis=dict(**_AXIS_STYLE, title=y_title),
    )


def zone_bar_chart(zone_frame: pd.DataFrame, height: int = 260) -> go.Figure:
    """九宫格区域缺陷数量柱状图。

    使用单一品牌蓝，不用红黄渐变——红色专属于缺陷告警语义，
    区域密度已由九宫格 tile 的深浅表达。

    Args:
        zone_frame: 含 `区域` 与 `缺陷数量` 两列的 DataFrame。
        height: 图表高度（像素）。

    Returns:
        配置好的 Plotly 图对象。
    """
    figure = go.Figure(
        go.Bar(
            x=zone_frame["区域"],
            y=zone_frame["缺陷数量"],
            marker=dict(
                color=COLORS["brand"],
                line=dict(color=COLORS["brand_strong"], width=0),
            ),
            text=zone_frame["缺陷数量"],
            textposition="outside",
            textfont=dict(size=12, color=COLORS["text"], family=_FONT_FAMILY),
            cliponaxis=False,
            hovertemplate="区域 %{x}<br>缺陷 %{y} 个<extra></extra>",
        )
    )
    figure.update_layout(**_base_layout(height, "区域", "缺陷数量（个）"))
    figure.update_yaxes(rangemode="tozero", nticks=5)
    return figure


def class_bar_chart(class_frame: pd.DataFrame, height: int = 260) -> go.Figure:
    """缺陷类别数量横向条形图。

    类别名是中文词组，横向条形更易读，也不会挤在一起。

    Args:
        class_frame: 含 `缺陷类别` 与 `数量` 两列的 DataFrame。
        height: 图表高度（像素）。

    Returns:
        配置好的 Plotly 图对象。
    """
    figure = go.Figure(
        go.Bar(
            x=class_frame["数量"],
            y=class_frame["缺陷类别"],
            orientation="h",
            marker=dict(color=COLORS["brand"], line=dict(width=0)),
            text=class_frame["数量"],
            textposition="outside",
            textfont=dict(size=12, color=COLORS["text"], family=_FONT_FAMILY),
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x} 个<extra></extra>",
        )
    )
    layout = _base_layout(height, "数量（个）", "")
    layout["yaxis"] = dict(
        **_AXIS_STYLE,
        title="",
        autorange="reversed",
    )
    figure.update_layout(**layout)
    figure.update_xaxes(rangemode="tozero", nticks=5)
    return figure
