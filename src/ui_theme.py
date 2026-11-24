"""UI 主题与可复用组件。

工业质检仪表盘的设计系统落地层。所有视觉 token 集中在 `THEME_CSS`，
页面组件通过 `inject_theme()` 一次性注入，避免样式散落在各处。

设计依据（ui-ux-pro-max 引擎推导 + 人工校正）：
- 产品类型：Dashboard / Admin
- 风格：Data / Dashboard + Industrial / Utility
- 语义色规范：green=success、red=error、amber=warn、blue=info（UX-037）
- 单一品牌强调色，克制使用（UX-039）
- 状态不只靠颜色表达（UX-038 / UX-003）

引擎推荐的 Calm Analogous 001 为红棕主色，但红色在质检语义中
专属于"缺陷/报警"，作主色会与缺陷标注冲突，故校正为工业冷蓝主色。
"""

from __future__ import annotations

import streamlit as st

from src.zone_analysis import ZoneAnalysis

# ---------------------------------------------------------------- 设计 token

COLORS = {
    "brand": "#185FA5",
    "brand_strong": "#134E86",
    "brand_soft": "#EAF2FB",
    "accent": "#0F6E56",
    "accent_soft": "#E6F4EF",
    "bg": "#F6F8FB",
    "surface": "#FFFFFF",
    "surface_2": "#EFF3F8",
    "border": "#DFE6EF",
    "border_strong": "#C9D4E2",
    "text": "#14202E",
    "text_2": "#5B6B7C",
    "text_3": "#8A98A8",
    "success": "#15803D",
    "success_soft": "#EAF7EF",
    "warning": "#B45309",
    "warning_soft": "#FEF6E7",
    "danger": "#B91C1C",
    "danger_soft": "#FDECEC",
    "info": "#185FA5",
    "info_soft": "#EAF2FB",
}

# 置信度 → 严重度（缺陷框与统计表共用的语义分级）
SEVERITY_COLORS = {
    "high": COLORS["danger"],
    "medium": COLORS["warning"],
    "low": "#64748B",
}

SEVERITY_LABELS = {
    "high": "高置信",
    "medium": "中置信",
    "low": "低置信",
}

_RADIUS_CARD = "12px"
_RADIUS_CONTROL = "8px"
_SPACE = {"1": "4px", "2": "8px", "3": "12px", "4": "16px", "5": "24px", "6": "32px"}
_FONT_STACK = (
    '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", '
    '"PingFang SC", "Noto Sans SC", "Helvetica Neue", Arial, sans-serif'
)
_MONO_STACK = '"JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace'

THEME_CSS = f"""
<style>
:root {{
  --brand: {COLORS["brand"]};
  --brand-strong: {COLORS["brand_strong"]};
  --brand-soft: {COLORS["brand_soft"]};
  --accent: {COLORS["accent"]};
  --accent-soft: {COLORS["accent_soft"]};
  --bg: {COLORS["bg"]};
  --surface: {COLORS["surface"]};
  --surface-2: {COLORS["surface_2"]};
  --border: {COLORS["border"]};
  --border-strong: {COLORS["border_strong"]};
  --text: {COLORS["text"]};
  --text-2: {COLORS["text_2"]};
  --text-3: {COLORS["text_3"]};
  --success: {COLORS["success"]};
  --success-soft: {COLORS["success_soft"]};
  --warning: {COLORS["warning"]};
  --warning-soft: {COLORS["warning_soft"]};
  --danger: {COLORS["danger"]};
  --danger-soft: {COLORS["danger_soft"]};
  --info: {COLORS["info"]};
  --info-soft: {COLORS["info_soft"]};
  --radius-card: {_RADIUS_CARD};
  --radius-control: {_RADIUS_CONTROL};
  --space-1: {_SPACE["1"]};
  --space-2: {_SPACE["2"]};
  --space-3: {_SPACE["3"]};
  --space-4: {_SPACE["4"]};
  --space-5: {_SPACE["5"]};
  --space-6: {_SPACE["6"]};
  --font-sans: {_FONT_STACK};
  --font-mono: {_MONO_STACK};
  --shadow-card: 0 1px 2px rgba(20, 32, 46, 0.06), 0 1px 3px rgba(20, 32, 46, 0.04);
}}

html, body, [class*="css"] {{
  font-family: var(--font-sans);
  color: var(--text);
}}

.stApp {{ background: var(--bg); }}

.block-container {{
  padding-top: 2rem;
  padding-bottom: 3rem;
  max-width: 1400px;
}}

/* Streamlit 默认标题字号偏大，需提高特异性覆盖 */
h1,
[data-testid="stHeading"] h1,
[data-testid="stHeadingWithActionElements"] h1 {{
  color: var(--text);
  font-size: 25px !important;
  line-height: 1.45 !important;
  font-weight: 600 !important;
  letter-spacing: -0.02em;
  padding: 0.06em 0 0.1em 0 !important;
}}
h2,
[data-testid="stHeading"] h2,
[data-testid="stHeadingWithActionElements"] h2 {{
  color: var(--text);
  font-size: 18px !important;
  line-height: 1.5 !important;
  font-weight: 600 !important;
  padding: 0.04em 0 0.08em 0 !important;
}}
h3,
[data-testid="stHeading"] h3,
[data-testid="stHeadingWithActionElements"] h3 {{
  color: var(--text);
  font-size: 15px !important;
  line-height: 1.5 !important;
  font-weight: 600 !important;
  padding: 0.04em 0 0.08em 0 !important;
}}

/* 隐藏 Streamlit 默认工具栏与页脚菜单，保持界面专注 */
[data-testid="stToolbar"] {{ display: none; }}
#MainMenu {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}

p, li, .stMarkdown {{ font-size: 14px; line-height: 1.65; }}

[data-testid="stCaptionContainer"], .stCaption {{ color: var(--text-2); font-size: 12px; }}

[data-testid="stSidebar"] {{
  background: var(--surface);
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}

[data-testid="stMetric"] {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: var(--space-4);
  box-shadow: var(--shadow-card);
}}
[data-testid="stMetricLabel"] p {{
  color: var(--text-2);
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.02em;
}}
[data-testid="stMetricValue"] {{
  color: var(--text);
  font-size: 28px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}}

[data-testid="stImage"] img {{
  border-radius: var(--radius-card);
  border: 1px solid var(--border);
  background: var(--surface);
}}

[data-testid="stDataFrame"] {{
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  overflow: hidden;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: var(--space-1); border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{
  height: 40px;
  padding: 0 var(--space-4);
  font-size: 14px;
  color: var(--text-2);
  border-radius: var(--radius-control) var(--radius-control) 0 0;
}}
.stTabs [aria-selected="true"] {{ color: var(--brand); font-weight: 600; background: var(--brand-soft); }}

.stButton > button, .stDownloadButton > button {{
  border-radius: var(--radius-control);
  font-weight: 500;
  font-size: 14px;
  min-height: 40px;
  border: 1px solid var(--border-strong);
  transition: background-color 120ms ease, border-color 120ms ease;
}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background: var(--brand);
  border-color: var(--brand);
  color: #FFFFFF;
}}
.stButton > button[kind="primary"]:hover {{ background: var(--brand-strong); border-color: var(--brand-strong); }}
.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible,
.stSelectbox [role="combobox"]:focus-visible,
[data-testid="stFileUploader"] button:focus-visible {{
  outline: 2px solid var(--brand);
  outline-offset: 2px;
}}

[data-testid="stFileUploader"] section {{
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius-card);
  background: var(--surface);
  padding: var(--space-3);
}}
[data-testid="stFileUploader"] section:hover {{ border-color: var(--brand); background: var(--brand-soft); }}

[data-testid="stExpander"] {{
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  background: var(--surface);
  overflow: hidden;
}}
[data-testid="stExpander"] summary {{ font-size: 14px; font-weight: 500; }}

[data-testid="stAlert"] {{ border-radius: var(--radius-card); border: 1px solid var(--border); }}

hr {{ border-color: var(--border); margin: var(--space-5) 0; }}

/* ---------------------------------------------------- 自定义组件 */

.sv-hero {{
  background: linear-gradient(180deg, var(--surface) 0%, var(--bg) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: var(--space-5) var(--space-6);
  margin-bottom: var(--space-4);
}}
.sv-hero-title {{
  display: flex;
  align-items: center;
  gap: var(--space-3);
  font-size: 24px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.02em;
  margin: 0 0 var(--space-2) 0;
}}
.sv-hero-sub {{ font-size: 13px; color: var(--text-2); margin: 0 0 var(--space-4) 0; line-height: 1.6; }}
.sv-hero-chips {{ display: flex; flex-wrap: wrap; gap: var(--space-2); }}

.sv-chip {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text-2);
  white-space: nowrap;
}}
.sv-chip-dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex: 0 0 auto; }}
.sv-chip.is-success {{ background: var(--success-soft); border-color: #BFE3CB; color: var(--success); }}
.sv-chip.is-warning {{ background: var(--warning-soft); border-color: #F0D9AE; color: var(--warning); }}
.sv-chip.is-danger {{ background: var(--danger-soft); border-color: #F3C4C4; color: var(--danger); }}
.sv-chip.is-info {{ background: var(--info-soft); border-color: #C6DBF3; color: var(--info); }}

.sv-section {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--space-3);
  margin: var(--space-6) 0 var(--space-3) 0;
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--border);
}}
.sv-section-title {{
  font-size: 16px;
  font-weight: 600;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: var(--space-2);
}}
.sv-section-caption {{ font-size: 12px; color: var(--text-3); }}

.sv-kpi {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--space-3); }}
.sv-kpi-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: var(--space-4);
  box-shadow: var(--shadow-card);
}}
.sv-kpi-label {{ font-size: 12px; color: var(--text-2); font-weight: 500; margin-bottom: var(--space-2); }}
.sv-kpi-value {{
  font-size: 30px;
  font-weight: 600;
  line-height: 1.1;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}}
.sv-kpi-hint {{ font-size: 12px; color: var(--text-3); margin-top: var(--space-2); }}

.sv-empty {{
  background: var(--surface);
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius-card);
  padding: var(--space-6) var(--space-5);
  text-align: center;
  color: var(--text-2);
}}
.sv-empty-title {{ font-size: 15px; font-weight: 600; color: var(--text); margin-bottom: var(--space-2); }}
.sv-empty-body {{ font-size: 13px; line-height: 1.7; margin-bottom: 0; }}
.sv-empty-steps {{ display: flex; justify-content: center; gap: var(--space-4); flex-wrap: wrap; margin-top: var(--space-4); }}
.sv-empty-step {{ font-size: 12px; color: var(--text-2); display: flex; align-items: center; gap: 6px; }}
.sv-empty-step-num {{
  width: 18px; height: 18px; border-radius: 50%;
  background: var(--brand-soft); color: var(--brand);
  font-size: 11px; font-weight: 600;
  display: inline-flex; align-items: center; justify-content: center;
}}

.sv-zone-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--space-2); }}
.sv-zone-tile {{
  position: relative;
  border: 1px solid var(--border);
  border-radius: var(--radius-control);
  padding: var(--space-3);
  min-height: 84px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  background: var(--surface);
}}
.sv-zone-tile.is-hotspot {{ border-color: var(--danger); box-shadow: inset 0 0 0 1px var(--danger); }}
.sv-zone-name {{ font-size: 12px; font-weight: 500; color: var(--text-2); }}
.sv-zone-count {{
  font-size: 24px;
  font-weight: 600;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  line-height: 1;
}}
.sv-zone-flag {{
  position: absolute;
  top: 8px; right: 8px;
  font-size: 11px;
  font-weight: 600;
  color: var(--danger);
  background: var(--danger-soft);
  border-radius: 999px;
  padding: 2px 6px;
}}
.sv-zone-legend {{
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-top: var(--space-3);
  font-size: 12px;
  color: var(--text-3);
}}
.sv-zone-legend-bar {{
  height: 8px;
  flex: 1;
  border-radius: 999px;
  background: linear-gradient(90deg, #F1F4F8 0%, #F5C4B3 55%, #B91C1C 100%);
  border: 1px solid var(--border);
}}

.sv-rule {{
  background: var(--surface-2);
  border-left: 3px solid var(--brand);
  border-radius: 0 var(--radius-control) var(--radius-control) 0;
  padding: var(--space-3) var(--space-4);
  font-size: 12px;
  color: var(--text-2);
  line-height: 1.7;
}}

@media (max-width: 1100px) {{
  .sv-kpi {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
@media (max-width: 720px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  .sv-kpi {{ grid-template-columns: minmax(0, 1fr); }}
  .sv-hero {{ padding: var(--space-4); }}
  .sv-hero-title {{ font-size: 19px; }}
  .sv-zone-tile {{ min-height: 72px; }}
}}
</style>
"""


def inject_theme() -> None:
    """注入全局主题样式。每个页面调用一次即可。"""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- 语义分级


def severity_level(confidence: float) -> str:
    """把置信度映射为严重度等级。

    Args:
        confidence: 检测置信度，0–1。

    Returns:
        `"high"`（>=0.70）、`"medium"`（>=0.40）或 `"low"`。
    """
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.40:
        return "medium"
    return "low"


def severity_color(confidence: float) -> str:
    """返回置信度对应的语义色值。"""
    return SEVERITY_COLORS[severity_level(confidence)]


# ---------------------------------------------------------------- 组件


def hero(title: str, subtitle: str, chips: list[tuple[str, str]] | None = None) -> None:
    """渲染顶部标题区（HTML 卡片形态）。

    Args:
        title: 主标题。
        subtitle: 一句话说明。
        chips: 状态徽章列表，每项为 `(文本, 语义)`，
            语义取 `success` / `warning` / `danger` / `info` / `neutral`。
    """
    chip_html = _chips_html(chips)

    st.markdown(
        f"""
        <div class="sv-hero">
          <p class="sv-hero-title">{title}</p>
          <p class="sv-hero-sub">{subtitle}</p>
          <div class="sv-hero-chips">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chips_html(chips: list[tuple[str, str]] | None) -> str:
    html = ""
    for text, tone in chips or []:
        css_class = f" is-{tone}" if tone and tone != "neutral" else ""
        html += f'<span class="sv-chip{css_class}"><span class="sv-chip-dot"></span>{text}</span>'
    return html


def hero_chips(chips: list[tuple[str, str]]) -> None:
    """只渲染一行状态徽章，配合 `st.title` 使用。

    保留 `st.title` 是为了让页面拥有真正的 h1 语义（无障碍与测试可查），
    视觉层由 CSS 统一控制；徽章作为补充信息单独成行。
    """
    st.markdown(
        f'<div class="sv-hero-chips" style="margin-bottom:8px">{_chips_html(chips)}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "", icon: str = "") -> None:
    """渲染带下划线的分区标题。"""
    icon_html = f"{icon} " if icon else ""
    caption_html = f'<span class="sv-section-caption">{caption}</span>' if caption else ""
    st.markdown(
        f"""
        <div class="sv-section">
          <span class="sv-section-title">{icon_html}{title}</span>
          {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    """渲染一行 KPI 卡片。

    Args:
        items: 每项为 `(标签, 数值, 说明)`，说明可为空串。
    """
    cards = ""
    for label, value, hint in items:
        hint_html = f'<div class="sv-kpi-hint">{hint}</div>' if hint else ""
        cards += (
            f'<div class="sv-kpi-card">'
            f'<div class="sv-kpi-label">{label}</div>'
            f'<div class="sv-kpi-value">{value}</div>'
            f"{hint_html}</div>"
        )
    st.markdown(f'<div class="sv-kpi">{cards}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str, steps: list[str] | None = None) -> None:
    """渲染空状态引导卡，避免页面出现无意义空白。"""
    steps_html = ""
    if steps:
        inner = "".join(
            f'<span class="sv-empty-step">'
            f'<span class="sv-empty-step-num">{index}</span>{text}</span>'
            for index, text in enumerate(steps, start=1)
        )
        steps_html = f'<div class="sv-empty-steps">{inner}</div>'

    st.markdown(
        f"""
        <div class="sv-empty">
          <div class="sv-empty-title">{title}</div>
          <p class="sv-empty-body">{body}</p>
          {steps_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def rule_note(text: str) -> None:
    """渲染规则说明条（浅底 + 左侧强调线）。"""
    st.markdown(f'<div class="sv-rule">{text}</div>', unsafe_allow_html=True)


def build_zone_tiles_html(analysis: ZoneAnalysis) -> str:
    """把九宫格区域统计渲染为热力 tile。

    颜色深浅表示该区域缺陷数量，热点区域额外加边框与文字标记，
    不只依赖颜色传达信息。

    Args:
        analysis: `analyze_zones` 的返回值。

    Returns:
        可直接交给 `st.markdown(unsafe_allow_html=True)` 的 HTML 片段。
    """
    max_count = max(analysis.counts.values()) if analysis.counts else 0

    tiles = ""
    for name, count in analysis.counts.items():
        ratio = (count / max_count) if max_count else 0.0
        # 由浅到深：0% 接近表面色，100% 用告警红
        alpha = 0.08 + 0.55 * ratio if count else 0.0
        is_hotspot = analysis.hotspot is not None and name == analysis.hotspot
        tile_class = "sv-zone-tile is-hotspot" if is_hotspot else "sv-zone-tile"
        background = (
            f"background: rgba(185, 28, 28, {alpha:.3f});" if count else ""
        )
        flag = '<span class="sv-zone-flag">集中</span>' if is_hotspot else ""

        tiles += (
            f'<div class="{tile_class}" style="{background}">'
            f"{flag}"
            f'<span class="sv-zone-name">{name}</span>'
            f'<span class="sv-zone-count">{count}</span>'
            f"</div>"
        )

    return (
        f'<div class="sv-zone-grid">{tiles}</div>'
        f'<div class="sv-zone-legend">'
        f"<span>少</span>"
        f'<span class="sv-zone-legend-bar"></span>'
        f"<span>多</span>"
        f'<span style="margin-left:auto">数字为该区域缺陷数量，红框为缺陷集中区域</span>'
        f"</div>"
    )
