"""Streamlit Web 主页面。

基于 Supervision 的钢材表面缺陷智能检测与区域质量分析系统。

页面结构对应项目方案书第 7 章：
- 顶部标题区：系统名称、项目说明、当前模型信息
- 左侧参数区：图片上传、置信度阈值、开始检测按钮
- 图片展示区：原始图片、Supervision 标注结果图
- 缺陷统计区：缺陷总数、类别统计、置信度明细
- 区域分析区：3×3 区域数量、缺陷集中区域图表
- 操作区：下载标注图片、导出检测明细
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import supervision as sv

from src.annotation import annotate_image
from src.chinese_label import find_chinese_font
from src.detection import DEFAULT_CLASS_NAMES, DetectionError, ModelNotFoundError, detect_image, load_model
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
    page_icon="🔩",
    layout="wide",
)


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
        cv2.polylines(canvas, [points], True, (80, 80, 80), 1)

    font = find_chinese_font(16)
    for index, name in enumerate(ZONE_NAMES):
        row, col = divmod(index, 3)
        x = int((col + 0.5) * width / 3)
        y = int((row + 0.5) * height / 3)

        if font is None:
            cv2.putText(
                canvas, name, (x - 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2
            )
            continue

        from PIL import Image, ImageDraw

        pil_canvas = Image.fromarray(canvas)
        drawer = ImageDraw.Draw(pil_canvas)
        drawer.text((x - 26, y - 10), name, font=font, fill=(60, 60, 60))
        canvas = np.array(pil_canvas)[:, :, ::-1].copy()

    return canvas


def render_sidebar():
    """渲染左侧参数区。"""
    with st.sidebar:
        st.header("检测参数")

        uploaded = st.file_uploader(
            "上传钢材表面图片",
            type=["jpg", "jpeg", "png"],
            help=f"支持格式：{'、'.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

        conf_threshold = st.slider(
            "置信度阈值",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="阈值越低检出越多缺陷，也更容易出现误检",
        )

        show_grid = st.checkbox("在原图上显示九宫格分割线", value=True)

        st.divider()
        st.subheader("当前模型")

        if WEIGHTS_PATH.exists():
            st.success("已就绪")
            st.caption(f"权重文件：{WEIGHTS_PATH.name}")
            st.caption(f"文件大小：{WEIGHTS_PATH.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            st.error("模型缺失")
            st.caption("请先运行 python train.py 训练模型")

        return uploaded, conf_threshold, show_grid


def render_header():
    """渲染顶部标题区。"""
    st.title("🔩 基于 Supervision 的钢材表面缺陷智能检测与区域质量分析系统")
    st.caption(
        "YOLO 负责识别缺陷类别与位置，Supervision 负责检测结果统一处理、框与标签标注，"
        "并通过 PolygonZone 完成 3×3 区域缺陷分布统计。"
    )
    st.info(
        "本系统是基于公开数据集的验证原型，检测结果用于辅助人工复检，"
        "不作为实际产线的质量判定依据。"
    )


def render_images(original: np.ndarray, annotated: np.ndarray, grid: np.ndarray, show_grid: bool):
    """并排展示原图与标注结果图。"""
    left, right = st.columns(2)
    with left:
        st.subheader("原始图片")
        st.image(grid if show_grid else original, use_container_width=True)
    with right:
        st.subheader("Supervision 标注结果")
        st.image(annotated, use_container_width=True)


def render_statistics(detections: sv.Detections, class_names: list[str]):
    """渲染缺陷统计区。"""
    summary = summarize_detections(detections, class_names)

    metric_columns = st.columns(3)
    metric_columns[0].metric("检出缺陷总数", summary["total"])
    metric_columns[1].metric("缺陷类别数", len(summary["class_counts"]))
    metric_columns[2].metric("平均置信度", f"{summary['mean_confidence']:.3f}")

    if summary["total"] == 0:
        st.warning(
            "未检出超过当前阈值的缺陷。注意：这不等于产品一定合格，仍需人工复核。",
            icon="⚠️",
        )
        return summary

    st.subheader("各类缺陷数量")
    counts_frame = pd.DataFrame(
        {"缺陷类别": list(summary["class_counts"].keys()), "数量": list(summary["class_counts"].values())}
    )
    st.bar_chart(counts_frame.set_index("缺陷类别"), height=260)

    st.subheader("缺陷明细")
    records = detections_to_records(detections, class_names)
    table = pd.DataFrame(records)
    table["置信度"] = table["置信度"].map(lambda value: f"{value:.3f}")
    for column in ["x1", "y1", "x2", "y2", "宽度", "高度"]:
        table[column] = table[column].map(lambda value: f"{value:.1f}")
    st.dataframe(table, use_container_width=True, hide_index=True)

    return summary


def render_zone_analysis(detections: sv.Detections, width: int, height: int):
    """渲染 3×3 区域分析区。"""
    st.subheader("3×3 区域缺陷分布分析")

    analysis = analyze_zones(detections, width=width, height=height)

    zone_frame = pd.DataFrame(
        {
            "区域": list(analysis.counts.keys()),
            "缺陷数量": list(analysis.counts.values()),
        }
    )

    chart = px.bar(
        zone_frame,
        x="区域",
        y="缺陷数量",
        color="缺陷数量",
        color_continuous_scale="YlOrRd",
        text="缺陷数量",
    )
    chart.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(chart, use_container_width=True)

    if analysis.hotspot is None:
        st.info("本次未检出缺陷，无法判断缺陷集中区域。")
    else:
        st.success(
            f"缺陷集中区域：**{analysis.hotspot}**，该区域检出 {analysis.hotspot_count} 个缺陷。"
        )

    st.caption(
        f"区域计数合计 {analysis.total} 个，参与统计的检测目标 {analysis.assigned_count} 个，"
        f"未归属（中心点在图像外）{analysis.unassigned_count} 个。"
        "锚点统一取检测框中心点，边界上的目标按从上到下、从左到右的顺序唯一归属。"
    )

    with st.expander("各区域的缺陷类别构成"):
        rows = []
        for zone_name in ZONE_NAMES:
            for class_id, count in analysis.class_counts.get(zone_name, {}).items():
                rows.append(
                    {
                        "区域": zone_name,
                        "缺陷类别": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                        "数量": count,
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无数据。")

    return analysis


def render_downloads(annotated: np.ndarray, detections: sv.Detections, class_names: list[str]):
    """渲染结果下载区。"""
    st.subheader("结果导出")
    columns = st.columns(2)

    with columns[0]:
        st.download_button(
            "下载标注图片 (PNG)",
            data=encode_png_bytes(annotated),
            file_name="annotated_result.png",
            mime="image/png",
            use_container_width=True,
        )

    with columns[1]:
        records = detections_to_records(detections, class_names)
        csv_frame = pd.DataFrame(records)
        st.download_button(
            "导出检测明细 (CSV)",
            data=csv_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="detections.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    render_header()
    uploaded, conf_threshold, show_grid = render_sidebar()

    demo_candidates = sorted((PROJECT_ROOT / "samples").glob("*.jpg")) if (PROJECT_ROOT / "samples").exists() else []
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
        st.info("请在左侧上传一张钢材表面图片，或展开上方使用示例图片。")
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
        st.error(f"图片读取失败：{error}")
        return

    if not st.button("开始检测", type="primary", use_container_width=True):
        st.caption("点击上方按钮开始检测。")
        return

    if not WEIGHTS_PATH.exists():
        st.error(
            "未找到模型权重文件 weights/best.pt，检测无法执行。"
            "请先运行 python train.py 完成模型训练。"
        )
        return

    try:
        with st.spinner("正在加载模型…"):
            model = get_model(str(WEIGHTS_PATH))
    except (ModelNotFoundError, DetectionError) as error:
        st.error(f"模型加载失败：{error}")
        return

    height, width = image.shape[:2]

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
        st.error(f"检测失败：{error}")
        return

    st.caption(f"图片尺寸 {width}×{height} 像素，检测耗时 {elapsed:.0f} 毫秒（CPU 推理）")

    annotated = annotate_image(image, detections, DEFAULT_CLASS_NAMES)
    grid_image = draw_zone_grid(image) if show_grid else image

    render_images(image, annotated, grid_image, show_grid)
    render_statistics(detections, DEFAULT_CLASS_NAMES)
    render_zone_analysis(detections, width, height)
    render_downloads(annotated, detections, DEFAULT_CLASS_NAMES)


if __name__ == "__main__":
    main()
