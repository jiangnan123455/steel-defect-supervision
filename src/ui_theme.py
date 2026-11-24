"""UI 主题与可复用组件。

工业质检工作台的设计系统落地层。所有视觉 token 集中在 `THEME_CSS`，
页面组件通过 `inject_theme()` 一次性注入，避免样式散落在各处。

版面结构（重排后）：

    顶栏            产品名 + 一句话说明                状态徽章
    工作台          画布（图像为主）                   控制轨（01 选图 / 02 条件 / 03 执行）
    结果带          指标条 → 区域分布 | 类别构成 → 明细表 + 导出
    页脚            数据来源                            边界声明

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
    "brand_strong": "#124B84",
    "brand_soft": "#EAF2FB",
    "accent": "#0F6E56",
    "accent_soft": "#E6F4EF",
    "bg": "#F4F6FA",
    "surface": "#FFFFFF",
    "surface_2": "#F1F4F9",
    "border": "#DDE4EE",
    "border_strong": "#C6D2E1",
    "text": "#131E2B",
    "text_2": "#56677A",
    "text_3": "#8494A6",
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

_RADIUS_CARD = "14px"
_RADIUS_CONTROL = "10px"
_SPACE = {"1": "4px", "2": "8px", "3": "12px", "4": "16px", "5": "24px", "6": "32px"}
_FONT_STACK = (
    '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", '
    '"Microsoft YaHei", "PingFang SC", "Noto Sans SC", "Helvetica Neue", Arial, sans-serif'
)
_MONO_STACK = '"JetBrains Mono", "Cascadia Mono", "SFMono-Regular", Consolas, monospace'

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
  --shadow-card: 0 1px 2px rgba(19, 30, 43, 0.05);
}}

/* ------------------------------------------------ 1. 画布、外壳与排版 */

html, body, [class*="css"] {{
  font-family: var(--font-sans);
  color: var(--text);
  -webkit-font-smoothing: antialiased;
}}

.stApp, [data-testid="stAppViewContainer"] {{ background: var(--bg); }}

/* 顶栏由页面自己绘制，Streamlit 的工具栏与侧栏全部让位 */
[data-testid="stHeader"] {{ display: none; }}
[data-testid="stToolbar"], #MainMenu {{ display: none; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}

.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {{
  max-width: 1380px !important;
  margin: 0 auto !important;
  padding: 24px 40px 48px !important;
}}

h1,
[data-testid="stHeading"] h1,
[data-testid="stHeadingWithActionElements"] h1 {{
  color: var(--text);
  font-size: 21px !important;
  line-height: 1.35 !important;
  font-weight: 600 !important;
  letter-spacing: -0.015em;
  padding: 0 !important;
}}
h2,
[data-testid="stHeading"] h2,
[data-testid="stHeadingWithActionElements"] h2 {{
  color: var(--text);
  font-size: 15px !important;
  line-height: 1.4 !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em;
  padding: 0 !important;
}}
h3,
[data-testid="stHeading"] h3,
[data-testid="stHeadingWithActionElements"] h3 {{
  color: var(--text);
  font-size: 14px !important;
  line-height: 1.45 !important;
  font-weight: 600 !important;
  padding: 0 !important;
}}

p, li, .stMarkdown {{ font-size: 14px; line-height: 1.62; }}
.stMarkdown p {{ margin-bottom: 0; }}

[data-testid="stCaptionContainer"] p,
.stCaption p,
[data-testid="stCaptionContainer"] {{
  color: var(--text-2);
  font-size: 12.5px;
  line-height: 1.5;
}}

[data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}

/* 顶栏：一条细线收住，标题与徽章同一基线 */
.st-key-topbar {{
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
  margin-bottom: 20px;
}}
.st-key-topbar h1 {{ font-size: 21px !important; margin-bottom: 3px !important; }}
.st-key-topbar [data-testid="stCaptionContainer"] {{ margin-top: 5px; }}
.st-key-topbar .sv-hero-chips {{ justify-content: flex-end; padding-bottom: 3px; }}

.sv-hero-chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.sv-chip {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  line-height: 1.4;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text-2);
  white-space: nowrap;
}}
.sv-chip-dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex: 0 0 auto; }}
.sv-chip.is-success {{ background: var(--success-soft); border-color: #BFE3CB; color: var(--success); }}
.sv-chip.is-warning {{ background: var(--warning-soft); border-color: #EFD8AC; color: var(--warning); }}
.sv-chip.is-danger {{ background: var(--danger-soft); border-color: #F2C3C3; color: var(--danger); }}
.sv-chip.is-info {{ background: var(--info-soft); border-color: #C6DBF3; color: var(--info); }}

/* ------------------------------------------------ 2. 工作台：画布与控制轨 */

.st-key-canvas {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: 14px;
  box-shadow: var(--shadow-card);
}}
/* 检视台：图片按原始比例居中，四周是浅底，不随容器拉伸 */
.sv-stage {{
  display: flex;
  align-items: center;
  justify-content: center;
  height: 432px;
  padding: 10px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface-2);
}}
.sv-stage img {{
  height: 100%;
  width: auto;
  max-width: 100%;
  object-fit: contain;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--surface);
}}

/* 未选图时画布放六类样例：3×2 的网格，铺满同一块高度 */
.sv-samples {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
.sv-samples figure {{ margin: 0; }}
.sv-samples img {{
  display: block;
  width: 100%;
  height: 188px;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--surface-2);
}}
.sv-samples figcaption {{
  margin-top: 6px;
  font-size: 12px;
  color: var(--text-2);
  text-align: center;
}}
/* 批量模式下失败的那一张，说明文字用告警色，不只靠颜色也有「失败」二字 */
.sv-samples figcaption.is-fail {{ color: var(--danger); }}

/* 画布右上角的待检测状态，与分段控件同一位置 */
p.sv-await {{
  margin: 0;
  text-align: right;
  font-size: 12px;
  color: var(--text-3);
}}

.st-key-rail {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  padding: 16px 16px 14px;
  box-shadow: var(--shadow-card);
}}
.st-key-rail [data-testid="stVerticalBlock"] {{ gap: 0.5rem; }}
.st-key-rail [data-testid="stAlert"] {{ padding: 7px 10px; }}
.st-key-rail [data-testid="stAlert"] p {{ font-size: 12px; line-height: 1.4; }}
.st-key-rail [data-testid="stWidgetLabel"] p {{ font-size: 12.5px; color: var(--text-2); }}

/* 列内小标题：与上方指标条拉开距离 */
[data-testid="stColumn"] h2,
[data-testid="stColumn"] h3 {{ margin: 16px 0 10px !important; }}

/* 明细带标题（与右对齐的导出按钮同一条基线） */
.sv-band-title {{
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--text);
}}

/* 画布上方的文件名与尺寸 */
p.sv-canvas-meta {{
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin: 0 0 4px;
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}}
.sv-canvas-meta span {{
  font-family: var(--font-mono);
  font-size: 11.5px;
  font-weight: 400;
  color: var(--text-3);
  font-variant-numeric: tabular-nums;
}}
.sv-canvas-meta em {{
  font-style: normal;
  margin-left: auto;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-2);
}}

/* 控制轨的分步编号 */
p.sv-step {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 2px 0 12px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--text-2);
}}
.sv-step i {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 22px;
  height: 20px;
  border-radius: 5px;
  background: var(--brand-soft);
  color: var(--brand);
  font-family: var(--font-mono);
  font-style: normal;
  font-size: 11px;
  font-weight: 600;
}}
.sv-hr {{ height: 1px; background: var(--border); margin: 19px 0 17px; }}

/* ------------------------------------------------ 3. 分区与指标条 */

.sv-section {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--space-3);
  margin: 28px 0 14px;
  padding-bottom: 9px;
  border-bottom: 1px solid var(--border);
}}
.sv-section-title {{
  display: inline-flex;
  align-items: center;
  gap: 9px;
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
}}
.sv-section-title::before {{
  content: "";
  width: 3px;
  height: 13px;
  border-radius: 2px;
  background: var(--brand);
}}
.sv-section-caption {{
  font-size: 12px;
  color: var(--text-3);
  font-variant-numeric: tabular-nums;
}}

.sv-ribbon {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  overflow: hidden;
  box-shadow: var(--shadow-card);
}}
.sv-ribbon > div {{ padding: 15px 18px 16px; }}
.sv-ribbon > div + div {{ border-left: 1px solid var(--border); }}
.sv-ribbon span {{ display: block; font-size: 12px; color: var(--text-2); }}
.sv-ribbon b {{
  display: block;
  margin-top: 6px;
  font-size: 27px;
  font-weight: 600;
  line-height: 1;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
  color: var(--text);
}}
.sv-ribbon small {{ display: block; margin-top: 6px; font-size: 11.5px; color: var(--text-3); }}
.sv-ribbon .is-warn b {{ color: var(--warning); }}

/* ------------------------------------------------ 4. 九宫格热力 */

.sv-zone-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
.sv-zone-tile {{
  position: relative;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 11px 13px 12px;
  min-height: 88px;
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
  top: 9px;
  right: 9px;
  font-size: 11px;
  font-weight: 600;
  color: var(--danger);
  background: var(--danger-soft);
  border-radius: 999px;
  padding: 2px 7px;
}}
.sv-zone-legend {{
  margin-top: 9px;
  font-size: 11.5px;
  line-height: 1.5;
  color: var(--text-3);
}}

/* ------------------------------------------------ 4b. 类别构成列表 */

.sv-class {{ display: grid; gap: 9px; }}
.sv-class-row {{
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr) 34px;
  align-items: center;
  gap: 12px;
}}
.sv-class-row span {{ font-size: 13px; color: var(--text-2); }}
.sv-class-row b {{
  font-size: 13px;
  font-weight: 600;
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--text);
}}
.sv-class-bar {{ height: 8px; border-radius: 999px; background: var(--surface-2); overflow: hidden; }}
.sv-class-bar i {{ display: block; height: 100%; border-radius: 999px; background: var(--brand); }}
.sv-class-foot {{
  display: flex;
  justify-content: space-between;
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--text-3);
}}

/* 置信度分级：三格一行，回答「模型有多确定」 */
.sv-grade {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-top: 18px;
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}}
.sv-grade > div {{ padding: 10px 12px; }}
.sv-grade > div + div {{ border-left: 1px solid var(--border); }}
.sv-grade span {{ display: block; font-size: 11.5px; color: var(--text-3); }}
.sv-grade b {{
  display: block;
  margin-top: 4px;
  font-size: 18px;
  font-weight: 600;
  line-height: 1;
  font-variant-numeric: tabular-nums;
  color: var(--text);
}}
p.sv-grade-title {{
  margin: 22px 0 9px;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-2);
}}

/* ------------------------------------------------ 5. 说明条与页脚 */

.sv-rule {{
  max-width: 96ch;
  background: var(--surface-2);
  border-left: 2px solid var(--brand);
  border-radius: 0 8px 8px 0;
  padding: 11px 14px;
  font-size: 12px;
  color: var(--text-2);
  line-height: 1.65;
}}

.sv-foot {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--space-5);
  margin-top: 30px;
  padding-top: 13px;
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--text-3);
}}
.sv-foot b {{ font-weight: 500; color: var(--text-2); }}

/* ------------------------------------------------ 6. 控件 */

.stButton > button,
.stDownloadButton > button {{
  border-radius: var(--radius-control);
  font-weight: 500;
  font-size: 13.5px;
  min-height: 38px;
  border: 1px solid var(--border-strong);
  background: var(--surface);
  color: var(--text);
  box-shadow: none;
  transition: background-color 120ms ease, border-color 120ms ease, transform 120ms ease;
}}
.stButton > button:hover,
.stDownloadButton > button:hover {{
  border-color: var(--brand);
  color: var(--brand);
  background: var(--brand-soft);
}}
.stButton > button:active,
.stDownloadButton > button:active {{ transform: translateY(1px); }}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {{
  background: var(--brand);
  border-color: var(--brand);
  color: #FFFFFF;
  min-height: 44px;
  font-size: 14px;
}}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {{
  background: var(--brand-strong);
  border-color: var(--brand-strong);
  color: #FFFFFF;
}}

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible,
.stSelectbox [role="combobox"]:focus-visible,
[data-testid="stFileUploader"] button:focus-visible,
[data-testid="stExpander"] summary:focus-visible {{
  outline: 2px solid var(--brand);
  outline-offset: 2px;
}}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {{
  border: 1px dashed var(--border-strong) !important;
  border-radius: var(--radius-control) !important;
  background: var(--surface-2) !important;
  padding: 12px !important;
  transition: border-color 120ms ease, background-color 120ms ease;
}}
[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploaderDropzone"]:hover {{
  border-color: var(--brand) !important;
  background: var(--brand-soft) !important;
}}
[data-testid="stFileUploader"] small {{ font-size: 11.5px; }}

[data-testid="stExpander"] {{
  border: 1px solid var(--border);
  border-radius: var(--radius-control);
  background: var(--surface-2);
  overflow: hidden;
}}
[data-testid="stExpander"] summary {{ font-size: 12.5px; font-weight: 500; padding: 7px 12px; }}
[data-testid="stExpander"] summary:hover {{ color: var(--brand); }}

[data-testid="stDataFrame"] {{
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}}
[data-testid="stDataFrame"] [role="columnheader"] {{
  background: var(--surface-2) !important;
  font-weight: 600;
}}

[data-testid="stAlert"] {{
  border-radius: 10px;
  border: 1px solid var(--border);
  font-size: 13px;
  padding: 10px 14px;
}}

/* 样例与视图切换：一组小胶囊，字号收到 12.5px 才排得进一列 */
.st-key-rail [data-testid="stButtonGroup"] button,
.st-key-canvas [data-testid="stButtonGroup"] button,
[data-testid="stPills"] button {{
  min-height: 29px;
  padding: 0 9px;
  font-size: 12.5px;
  line-height: 1.2;
  border-radius: 7px;
}}
.st-key-rail [data-testid="stButtonGroup"] {{ gap: 5px; flex-wrap: wrap; }}

hr {{ border-color: var(--border); margin: var(--space-5) 0; }}

/* ------------------------------------------------ 7. 窄屏收敛 */

@media (max-width: 1180px) {{
  .sv-ribbon {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .sv-ribbon > div:nth-child(3) {{ border-left: 0; }}
  .sv-ribbon > div:nth-child(n + 3) {{ border-top: 1px solid var(--border); }}
  .st-key-canvas [data-testid="stImage"] img {{ height: 360px; }}
}}
@media (max-width: 720px) {{
  .block-container,
  [data-testid="stMainBlockContainer"] {{ padding: 16px 16px 40px !important; }}
  .st-key-canvas [data-testid="stImage"] img {{ height: auto; width: 100%; }}
  .sv-zone-tile {{ min-height: 78px; }}
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


def _chips_html(chips: list[tuple[str, str]] | None) -> str:
    html = ""
    for text, tone in chips or []:
        css_class = f" is-{tone}" if tone and tone != "neutral" else ""
        html += f'<span class="sv-chip{css_class}"><span class="sv-chip-dot"></span>{text}</span>'
    return html


def hero_chips(chips: list[tuple[str, str]]) -> None:
    """渲染一行状态徽章，配合 `st.title` 使用。

    保留 `st.title` 是为了让页面拥有真正的 h1 语义（无障碍与测试可查），
    视觉层由 CSS 统一控制；徽章作为补充信息单独成行。
    """
    st.markdown(f'<div class="sv-hero-chips">{_chips_html(chips)}</div>', unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    """渲染分区标题：左侧色标 + 标题，右侧对齐的补充信息。"""
    caption_html = f'<span class="sv-section-caption">{caption}</span>' if caption else ""
    st.markdown(
        f'<div class="sv-section">'
        f'<span class="sv-section-title">{title}</span>'
        f"{caption_html}</div>",
        unsafe_allow_html=True,
    )


def step_label(index: int, text: str) -> None:
    """渲染控制轨的一步，编号用等宽字，形成 01/02/03 的阅读顺序。"""
    st.markdown(
        f'<p class="sv-step"><i>{index:02d}</i>{text}</p>', unsafe_allow_html=True
    )


def divider() -> None:
    """控制轨内的一条细分隔线（比 st.divider 更薄）。"""
    st.markdown('<div class="sv-hr"></div>', unsafe_allow_html=True)


def canvas_meta(filename: str, meta: str, trailing: str = "") -> None:
    """画布上方的文件名、尺寸与右侧状态。"""
    trailing_html = f"<em>{trailing}</em>" if trailing else ""
    st.markdown(
        f'<p class="sv-canvas-meta">{filename}<span>{meta}</span>{trailing_html}</p>',
        unsafe_allow_html=True,
    )


def stat_ribbon(items: list[tuple[str, str, str]]) -> None:
    """渲染一条四格指标带（不是四张浮动卡片）。

    Args:
        items: 每项为 `(标签, 数值, 说明)`，说明可为空串。
    """
    cells = ""
    for label, value, hint in items:
        warn_class = ' class="is-warn"' if label == "检出缺陷" and value == "0" else ""
        hint_html = f"<small>{hint}</small>" if hint else ""
        cells += (
            f"<div{warn_class}><span>{label}</span><b>{value}</b>{hint_html}</div>"
        )
    st.markdown(f'<div class="sv-ribbon">{cells}</div>', unsafe_allow_html=True)


def class_rows(counts: dict[str, int], total: int) -> str:
    """类别构成列表：名字 + 占比条 + 数量。

    只有一类时横向柱状图会退化成一条满格长条，列表在任何类别数下都成立。

    Args:
        counts: 类别名到数量的映射，调用方已按数量降序排好。
        total: 检出总数，用于算占比。

    Returns:
        可直接交给 `st.markdown(unsafe_allow_html=True)` 的 HTML 片段。
    """
    top = max(counts.values()) if counts else 0
    rows = ""
    for name, count in counts.items():
        width = (count / top * 100) if top else 0
        rows += (
            f'<div class="sv-class-row"><span>{name}</span>'
            f'<span class="sv-class-bar"><i style="width:{width:.1f}%"></i></span>'
            f"<b>{count}</b></div>"
        )
    share = f"{top / total * 100:.0f}%" if total else "0%"
    return (
        f'<div class="sv-class">{rows}</div>'
        f'<div class="sv-class-foot"><span>共 {len(counts)} 类 · {total} 个检出</span>'
        f"<span>最多一类占 {share}</span></div>"
    )


def grade_strip(counts: dict[str, int]) -> str:
    """置信度分级条：高 / 中 / 低三档各有多少个。

    Args:
        counts: 分级名到数量的映射，顺序即展示顺序。

    Returns:
        可直接交给 `st.markdown(unsafe_allow_html=True)` 的 HTML 片段。
    """
    cells = "".join(
        f"<div><span>{name}</span><b>{count}</b></div>" for name, count in counts.items()
    )
    return f'<div class="sv-grade">{cells}</div>'


def rule_note(text: str) -> None:
    """渲染规则说明条（浅底 + 左侧强调线）。"""
    st.markdown(f'<div class="sv-rule">{text}</div>', unsafe_allow_html=True)


def footer_note(left: str, right: str) -> None:
    """页脚一行：左侧数据来源，右侧边界声明。"""
    st.markdown(
        f'<div class="sv-foot"><span>{left}</span><span><b>{right}</b></span></div>',
        unsafe_allow_html=True,
    )


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
        # 底色只做 0–22% 的浅红，深色文字始终留在可读区间
        alpha = 0.04 + 0.18 * ratio if count else 0.0
        is_hotspot = analysis.hotspot is not None and name == analysis.hotspot
        tile_class = "sv-zone-tile is-hotspot" if is_hotspot else "sv-zone-tile"
        background = f"background: rgba(185, 28, 28, {alpha:.3f});" if count else ""
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
        f'<div class="sv-zone-legend">红色描边为缺陷集中区域，底色越深该区域检出越多</div>'
    )
