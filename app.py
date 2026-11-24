"""Streamlit Web 主页面。

基于 Supervision 的钢材表面缺陷智能检测与区域质量分析系统。

页面信息层级（对应项目方案书第 7 章）：
1. 顶部标题区：系统名称、一句话说明、状态徽章、边界声明
2. 左侧参数区：图片上传、置信度阈值、显示选项、模型状态
3. 结果概览：KPI 指标行
4. 视图对比：原始图（含九宫格）与 Supervision 标注结果
5. 缺陷构成：类别分布图与明细表
6. 区域分析：九宫格热力分布、区域柱状图、集中区域结论
7. 结果导出：标注图 PNG 与明细 CSV

视觉规范见 `src/ui_theme.py`，图表规范见 `src/ui_charts.py`。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import supervision as sv

from src.annotation import annotate_image
from src.chinese_label import find_chinese_font
from src.detection import (
    DEFAULT_CLASS_NAMES,
    DetectionError,
    ModelNotFoundError,
    detect_image,
    load_model,
)
from src.ui_charts import class_bar_chart, zone_bar_chart
from src.ui_theme import (
    COLORS,
    SEVERITY_LABELS,
    build_zone_tiles_html,
    empty_state,
    hero_chips,
    inject_theme,
    kpi_row,
    rule_note,
    severity_level,
    section,
)
from src.utils import (
    SUPPORTED_EXTENSIONS,
    decode_image_bytes,
    detections_to_records,
    encode_png_bytes,
    summarize_detections,
    validate_upload,
)
from src.zone_analysis import ZONE_NAMES, analyze_zones, build_grid_polygons

PROJECT_ROOT = Path(__file__).resolve().parent
# 允许通过环境变量覆盖权重路径，便于测试与临时切换模型
WEIGHTS_PATH = Path(
    os.environ.get("STEEL_WEIGHTS_PATH", PROJECT_ROOT / "weights" / "best.pt")
)

st.set_page_config(
    page_title="钢材表面缺陷检测与区域质量分析系统",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()


@st.cache_resource(show_spinner=False)
def get_model(weights_path: str):
    """加载并缓存 YOLO 模型，避免每次检测重新加载权重。"""
    return load_model(weights_path)


@st.cache_data(show_spinner=False)
def load_demo_image(path: str) -> bytes:
    return Path(path).read_bytes()


def draw_zone_grid(image: np.ndarray) -> np.ndarray:
    """在图像上绘制九宫格分割线与区域名称。"""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for polygon in build_grid_polygons(width, height):
        points = polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, (150, 160, 172), 1)

    font = find_chinese_font(15)
    for index, name in enumerate(ZONE_NAMES):
        row, col = divmod(index, 3)
        x = int((col + 0.5) * width / 3)
        y = int((row + 0.5) * height / 3)

        if font is None:
            cv2.putText(
                canvas, name, (x - 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 102, 116), 1
            )
            continue

        from PIL import Image, ImageDraw

        pil_canvas = Image.fromarray(canvas)
        drawer = ImageDraw.Draw(pil_canvas)
        # 白色描边保证在深色钢材纹理上依然可读
        drawer.text(
            (x - 26, y - 10),
            name,
            font=font,
            fill=(38, 50, 64),
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
        canvas = np.array(pil_canvas)[:, :, ::-1].copy()

    return canvas


def render_sidebar():
    """渲染左侧参数区，返回上传文件、阈值与显示选项。"""
    with st.sidebar:
        st.header("检测参数")
        st.caption("上传图片并设置置信度阈值后开始检测。")

        uploaded = st.file_uploader(
            "上传钢材表面图片",
            type=["jpg", "jpeg", "png"],
            help=f"支持格式：{'、'.join(sorted(SUPPORTED_EXTENSIONS))}，单张不超过 20 MB。",
        )

        conf_threshold = st.slider(
            "置信度阈值",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="阈值越低检出越多缺陷，也更容易出现误检；阈值越高越保守。",
        )

        show_grid = st.checkbox("在原图上显示九宫格分割线", value=True)

        st.divider()
        st.subheader("当前模型")

        if WEIGHTS_PATH.exists():
            st.success("已就绪")
            size_mb = WEIGHTS_PATH.stat().st_size / 1024 / 1024
            st.caption(f"权重文件：{WEIGHTS_PATH.name} · {size_mb:.1f} MB")
            st.caption("模型：YOLOv8n · 6 类钢材表面缺陷")
        else:
            st.error("模型缺失")
            st.caption("请先运行 `python train.py` 训练模型，或把 best.pt 放入 weights/ 目录。")

        st.divider()
        st.subheader("使用提示")
        st.caption("1. 优先使用清晰的正面拍摄图片。")
        st.caption("2. 漏检时适当降低阈值，误检偏多时提高阈值。")
        st.caption("3. 九宫格统计只描述缺陷分布，不代表质量等级。")

        return uploaded, conf_threshold, show_grid


def render_header():
    """渲染顶部标题区。"""
    st.title("基于 Supervision 的钢材表面缺陷智能检测与区域质量分析系统")

    chips = [("6 类缺陷", "neutral"), ("中心点锚点九宫格", "neutral")]
    if WEIGHTS_PATH.exists():
        chips.insert(0, ("模型已就绪", "success"))
    else:
        chips.insert(0, ("模型缺失", "danger"))
    hero_chips(chips)

    st.caption(
        "YOLO 负责识别缺陷类别与位置，Supervision 统一检测结果并完成框与标签标注，"
        "再通过 PolygonZone 输出 3×3 区域缺陷分布。"
    )
    rule_note(
        "本系统是基于公开数据集（NEU-DET）的验证原型。检测结果用于辅助人工复检，"
        "不作为实际产线的质量判定依据；<strong>「未检出缺陷」不等于产品合格</strong>。"
    )


def render_kpi(summary: dict, elapsed_ms: float):
    """渲染结果概览指标行。"""
    section("检测概览", caption=f"单张图片 · 耗时 {elapsed_ms:.0f} ms", icon="")
    kpi_row(
        [
            ("检出缺陷总数", str(summary["total"]), "当前阈值下的检出数量"),
            ("缺陷类别数", str(len(summary["class_counts"])), "共 6 类可识别缺陷"),
            ("平均置信度", f"{summary['mean_confidence']:.3f}", "所有检出目标的均值"),
        ]
    )


def render_images(
    original: np.ndarray,
    annotated: np.ndarray,
    grid: np.ndarray,
    show_grid: bool,
    width: int,
    height: int,
):
    """并排展示原图与 Supervision 标注结果。"""
    section("视图对比", caption=f"图片尺寸 {width}×{height} px")

    left, right = st.columns(2, gap="medium")
    with left:
        st.subheader("原始图片")
        st.image(
            grid if show_grid else original,
            use_container_width=True,
            caption="灰色线为 3×3 区域分割线" if show_grid else "未叠加分割线",
        )
    with right:
        st.subheader("Supervision 标注结果")
        st.image(
            annotated,
            use_container_width=True,
            caption="框与标签由 BoxAnnotator 与中文标签渲染器绘制",
        )


def render_statistics(detections: sv.Detections, class_names: list[str], summary: dict):
    """渲染缺陷构成区：类别分布图与明细表。"""
    section("缺陷构成")

    if summary["total"] == 0:
        st.warning(
            "未检出超过当前阈值的缺陷。注意：这不等于产品一定合格，仍需人工复核。",
            icon=":material/info:",
        )
        return

    chart_column, table_column = st.columns([1, 1.35], gap="medium")

    with chart_column:
        st.subheader("各类缺陷数量")
        class_frame = pd.DataFrame(
            {
                "缺陷类别": list(summary["class_counts"].keys()),
                "数量": list(summary["class_counts"].values()),
            }
        )
        st.plotly_chart(
            class_bar_chart(class_frame, height=min(300, max(140, 48 * len(class_frame) + 48))),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with table_column:
        st.subheader("缺陷明细")
        records = detections_to_records(detections, class_names)
        table = pd.DataFrame(records)
        table["置信度"] = table["置信度"].map(lambda value: f"{value:.3f}")
        for column in ["x1", "y1", "x2", "y2", "宽度", "高度"]:
            table[column] = table[column].map(lambda value: f"{value:.1f}")
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            height=min(420, max(170, 44 + 38 * len(table))),
        )
        st.caption(
            "置信度分级：高置信 ≥ 0.70 · 中置信 0.40–0.69 · 低置信 < 0.40"
        )


def render_zone_analysis(
    detections: sv.Detections,
    width: int,
    height: int,
    class_names: list[str],
):
    """渲染 3×3 区域分析区。"""
    section("3×3 区域缺陷分布分析", caption="锚点取检测框中心点")

    analysis = analyze_zones(detections, width=width, height=height)

    tiles_column, chart_column = st.columns([1, 1.25], gap="medium")

    with tiles_column:
        st.subheader("九宫格热力分布")
        st.markdown(build_zone_tiles_html(analysis), unsafe_allow_html=True)

    with chart_column:
        st.subheader("各区域缺陷数量")
        zone_frame = pd.DataFrame(
            {
                "区域": list(analysis.counts.keys()),
                "缺陷数量": list(analysis.counts.values()),
            }
        )
        st.plotly_chart(
            zone_bar_chart(zone_frame, height=280),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    if analysis.hotspot is None:
        st.info("本次未检出缺陷，无法判断缺陷集中区域。", icon=":material/info:")
    else:
        st.success(
            f"缺陷集中区域：**{analysis.hotspot}**，该区域检出 {analysis.hotspot_count} 个缺陷。"
            "建议优先复核该区域对应的工艺环节。",
            icon=":material/location_on:",
        )

    rule_note(
        f"区域计数合计 {analysis.total} 个，参与统计的检测目标 {analysis.assigned_count} 个，"
        f"未归属（中心点在图像外）{analysis.unassigned_count} 个。"
        "边界上的目标按从上到下、从左到右的顺序唯一归属，"
        "因此九个区域计数之和恒等于参与统计的检测目标数。"
    )

    with st.expander("查看各区域的缺陷类别构成"):
        rows = []
        for zone_name in ZONE_NAMES:
            for class_id, count in analysis.class_counts.get(zone_name, {}).items():
                rows.append(
                    {
                        "区域": zone_name,
                        "缺陷类别": (
                            class_names[class_id]
                            if class_id < len(class_names)
                            else f"class_{class_id}"
                        ),
                        "数量": count,
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("当前没有可展示的区域类别构成。")

    return analysis


def render_downloads(annotated: np.ndarray, detections: sv.Detections, class_names: list[str]):
    """渲染结果导出区。"""
    section("结果导出", caption="标注图用于复核，明细可用于留档")

    columns = st.columns(2, gap="medium")

    with columns[0]:
        st.download_button(
            "下载标注图片（PNG）",
            data=encode_png_bytes(annotated),
            file_name="annotated_result.png",
            mime="image/png",
            use_container_width=True,
        )

    with columns[1]:
        records = detections_to_records(detections, class_names)
        csv_frame = pd.DataFrame(records)
        st.download_button(
            "导出检测明细（CSV）",
            data=csv_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="detections.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_footer():
    """页脚：数据与边界说明。"""
    st.divider()
    st.caption(
        "数据集：NEU-DET 钢材表面缺陷数据集（清理修正版）· "
        "模型：YOLOv8n · 视觉处理：Supervision 0.32 · "
        "本系统为验证原型，不构成质量判定依据。"
    )


def main() -> None:
    render_header()
    uploaded, conf_threshold, show_grid = render_sidebar()

    demo_candidates = (
        sorted((PROJECT_ROOT / "samples").glob("*.jpg"))
        if (PROJECT_ROOT / "samples").exists()
        else []
    )
    if demo_candidates:
        with st.expander("没有图片？使用示例图片体验", expanded=uploaded is None):
            demo_choice = st.selectbox(
                "选择示例图片",
                [path.name for path in demo_candidates],
                index=0,
            )
            demo_path = next(path for path in demo_candidates if path.name == demo_choice)
            uploaded = None if uploaded is not None else demo_path

    if uploaded is None:
        empty_state(
            "还没有待检测的图片",
            "上传一张钢材表面图片，或展开上方的示例图片入口。"
            "系统会给出缺陷框、类别置信度与九宫格分布统计。",
            steps=["上传图片", "设置阈值", "开始检测"],
        )
        render_footer()
        return

    try:
        if isinstance(uploaded, (str, Path)):
            raw_bytes = uploaded.read_bytes()
            filename = uploaded.name
        else:
            raw_bytes = uploaded.getvalue()
            filename = uploaded.name
        validate_upload(filename, len(raw_bytes))
        image = decode_image_bytes(raw_bytes)
    except ValueError as error:
        st.error(f"图片读取失败：{error}", icon=":material/error:")
        render_footer()
        return

    height, width = image.shape[:2]

    # ?autorun=1 时跳过按钮等待，直接执行检测，便于演示与自动化截图
    autorun = st.query_params.get("autorun") in ("1", "true", "yes")

    action_column, meta_column = st.columns([1, 2], gap="medium")
    with action_column:
        start_clicked = st.button(
            "开始检测", type="primary", use_container_width=True
        )
    with meta_column:
        st.caption(
            f"已选择：{filename} · {width}×{height} px · 阈值 {conf_threshold:.2f}"
        )

    if not start_clicked and not autorun:
        st.caption("确认图片与阈值后，点击「开始检测」。")
        render_footer()
        return

    if not WEIGHTS_PATH.exists():
        st.error(
            "未找到模型权重文件 weights/best.pt，检测无法执行。"
            "请先运行 `python train.py` 完成模型训练。",
            icon=":material/error:",
        )
        render_footer()
        return

    try:
        with st.spinner("正在加载模型…"):
            model = get_model(str(WEIGHTS_PATH))
    except (ModelNotFoundError, DetectionError) as error:
        st.error(f"模型加载失败：{error}", icon=":material/error:")
        render_footer()
        return

    try:
        with st.spinner("正在检测缺陷…"):
            start = time.perf_counter()
            detections = detect_image(
                model,
                image,
                conf_threshold=conf_threshold,
                class_names=DEFAULT_CLASS_NAMES,
            )
            elapsed = (time.perf_counter() - start) * 1000
    except (DetectionError, ValueError) as error:
        st.error(f"检测失败：{error}", icon=":material/error:")
        render_footer()
        return

    summary = summarize_detections(detections, DEFAULT_CLASS_NAMES)
    annotated = annotate_image(image, detections, DEFAULT_CLASS_NAMES)
    grid_image = draw_zone_grid(image) if show_grid else image

    render_kpi(summary, elapsed)
    render_images(image, annotated, grid_image, show_grid, width, height)
    render_statistics(detections, DEFAULT_CLASS_NAMES, summary)
    render_zone_analysis(detections, width, height, DEFAULT_CLASS_NAMES)
    render_downloads(annotated, detections, DEFAULT_CLASS_NAMES)
    render_footer()


if __name__ == "__main__":
    main()
