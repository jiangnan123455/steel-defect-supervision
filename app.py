"""Streamlit Web 主页面。

基于 Supervision 的钢材表面缺陷智能检测与区域质量分析系统。

版面（重排后按「看过一张图 → 判定 → 复核」的顺序组织）：
1. 顶栏：产品名、一句话说明、状态徽章
2. 工作台左：画布（结果图为主，可切换标注结果 / 原图 / 九宫格）
3. 工作台右：控制轨（01 选择图片 → 02 判定条件 → 03 执行检测）
4. 结果带：指标条 → 区域分布 | 类别构成 → 缺陷明细与导出
5. 页脚：数据来源与边界声明

视觉规范见 `src/ui_theme.py`，图表规范见 `src/ui_charts.py`。
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import supervision as sv

from src.annotation import annotate_image
from src.batch import (
    MAX_BATCH_FILES,
    batch_csv,
    batch_frame,
    item_captions,
    run_batch,
)
from src.chinese_label import find_chinese_font
from src.detection import (
    DEFAULT_CLASS_NAMES,
    DetectionError,
    ModelNotFoundError,
    detect_image,
    load_model,
)
from src.ui_theme import (
    SEVERITY_LABELS,
    build_zone_tiles_html,
    canvas_meta,
    class_rows,
    grade_strip,
    divider,
    footer_note,
    hero_chips,
    inject_theme,
    rule_note,
    section,
    severity_level,
    stat_ribbon,
    step_label,
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

VIEWS = ("标注结果", "原图", "九宫格")
MODES = ("单张检测", "批量检测")

CLASS_SAMPLES = [
    ("crazing_101.jpg", "龟裂"),
    ("inclusion_102.jpg", "夹杂"),
    ("patches_109.jpg", "压痕斑块"),
    ("pitted_surface_10.jpg", "麻点"),
    ("rolled-in_scale_102.jpg", "氧化铁皮压入"),
    ("scratches_101.jpg", "划痕"),
]
SAMPLE_BY_NAME = {name: filename for filename, name in CLASS_SAMPLES}

st.set_page_config(
    page_title="钢材表面缺陷检测与区域质量分析系统",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_theme()


@st.cache_resource(show_spinner=False)
def get_model(weights_path: str):
    """加载并缓存 YOLO 模型，避免每次检测重新加载权重。"""
    return load_model(weights_path)


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


def zone_of_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    """按检测框中心点判断所属区域（与 `analyze_zones` 的归属规则一致）。"""
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    if not (0 <= center_x < width and 0 <= center_y < height):
        return "图外"
    column = min(2, max(0, int(center_x * 3 / width)))
    row = min(2, max(0, int(center_y * 3 / height)))
    return ZONE_NAMES[row * 3 + column]


# ---------------------------------------------------------------- 顶栏


def render_topbar() -> None:
    """顶栏：产品名在左，状态徽章在右，用一条细线收住。"""
    with st.container(key="topbar"):
        title_column, chip_column = st.columns([2.3, 1.2], vertical_alignment="center")
        with title_column:
            st.title("钢材表面缺陷检测与区域质量分析")
            st.caption(
                "YOLOv8n 识别缺陷类别与位置 · Supervision 统一标注 · 九宫格统计区域分布"
            )
        with chip_column:
            chips = [("6 类缺陷", "neutral"), ("中心点锚定九宫格", "neutral")]
            if WEIGHTS_PATH.exists():
                chips.insert(0, ("模型已就绪", "success"))
            else:
                chips.insert(0, ("模型缺失", "danger"))
            hero_chips(chips)


# ---------------------------------------------------------------- 控制轨


def render_rail() -> dict:
    """控制轨：模式 → 选图 → 定阈值 → 执行。

    Returns:
        页面状态字典：`mode` / `uploaded` / `sample_name` / `use_samples` /
        `conf_threshold` / `start_clicked`。
    """
    with st.container(key="rail"):
        # ?mode=batch 可直接进入批量模式，便于演示与截图
        default_mode = MODES[1] if st.query_params.get("mode") == "batch" else MODES[0]
        mode = st.segmented_control(
            "检测模式",
            list(MODES),
            default=default_mode,
            label_visibility="collapsed",
            key="mode_switch",
            help="单张：一次一张，可切换标注/原图/九宫格三视图；批量：一次多张，出汇总表与 CSV。",
        ) or default_mode
        st.caption(
            "单张模式看一张图的细节与区域分布；批量模式逐张跑完一批，出汇总与明细。"
        )

        divider()
        step_label(1, "选择图片")

        sample_name = None
        use_samples = False
        if mode == MODES[1]:
            uploaded = st.file_uploader(
                "上传钢材表面图片（可多选）",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                help=(
                    f"支持 {'、'.join(sorted(SUPPORTED_EXTENSIONS))}；"
                    f"单批最多 {MAX_BATCH_FILES} 张，单张不超过 20 MB。"
                ),
            )
            use_samples = st.toggle(
                "用六类样例组成一批（无需上传）",
                value=st.query_params.get("batch") == "samples",
                help="没有现成图片时，用 samples/ 下的六张样例演示整批流程。",
            )
        else:
            uploaded = st.file_uploader(
                "上传钢材表面图片",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                help=f"支持格式：{'、'.join(sorted(SUPPORTED_EXTENSIONS))}，单张不超过 20 MB。",
            )
            with st.expander(f"示例图片（{len(CLASS_SAMPLES)} 张）", expanded=False):
                # 默认载入第一张样例：页面一打开就有一张真实的钢材表面图，
                # 只需点一次「开始检测」；取消选择可回到六类样例库。
                # ?sample=压痕斑块 可直接指向某张样例，便于演示与截图
                default_sample = st.query_params.get("sample")
                if default_sample not in SAMPLE_BY_NAME:
                    default_sample = CLASS_SAMPLES[0][1]
                sample_name = st.pills(
                    "示例图片",
                    [name for _, name in CLASS_SAMPLES],
                    default=default_sample,
                    label_visibility="collapsed",
                    key="sample_pick",
                )

        divider()
        step_label(2, "判定条件")
        conf_threshold = st.slider(
            "置信度阈值",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="阈值越低检出越多缺陷，也更容易出现误检；阈值越高越保守。两种模式共用同一个阈值。",
        )

        divider()
        step_label(3, "执行检测")
        start_clicked = st.button(
            "开始批量检测" if mode == MODES[1] else "开始检测",
            type="primary",
            width="stretch",
        )

        if WEIGHTS_PATH.exists():
            size_mb = WEIGHTS_PATH.stat().st_size / 1024 / 1024
            st.success(f"模型已就绪 · YOLOv8n · {size_mb:.1f} MB", icon=":material/check_circle:")
        else:
            st.error("模型缺失", icon=":material/error:")
            st.caption("请先运行 `python train.py` 训练模型，或把 best.pt 放入 weights/ 目录。")

    return {
        "mode": mode,
        "uploaded": uploaded,
        "sample_name": sample_name,
        "use_samples": use_samples,
        "conf_threshold": conf_threshold,
        "start_clicked": start_clicked,
    }


# ---------------------------------------------------------------- 画布


def _data_uri(raw: bytes) -> str:
    """把图片字节包成 data URI：画布直接写 <img>，尺寸与留白完全可控。"""
    mime = "image/jpeg" if raw[:3] == bytes([0xFF, 0xD8, 0xFF]) else "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _stage(raw: bytes, alt: str) -> None:
    """检视台：图片按原始比例居中，四周留白，不随容器拉伸。"""
    st.markdown(
        f'<div class="sv-stage"><img src="{_data_uri(raw)}" alt="{alt}"></div>',
        unsafe_allow_html=True,
    )


def _sample_grid() -> str:
    """六类样例的 3×2 网格（未选图时的画布内容）。"""
    cells = ""
    for filename, name in CLASS_SAMPLES:
        path = PROJECT_ROOT / "samples" / filename
        if not path.exists():
            continue
        cells += (
            f'<figure><img src="{_data_uri(path.read_bytes())}" alt="{name}">'
            f"<figcaption>{name}</figcaption></figure>"
        )
    return f'<div class="sv-samples">{cells}</div>'


def render_canvas(
    filename: str | None,
    raw: bytes | None,
    image: np.ndarray | None,
    run: dict | None,
    conf_threshold: float,
    error: str | None,
) -> None:
    """画布：一张图占满面板，视图通过分段控件切换。"""
    with st.container(key="canvas"):
        if error:
            canvas_meta(filename or "无法读取", "—")
            st.error(f"图片读取失败：{error}", icon=":material/error:")
            return

        if image is None or raw is None:
            canvas_meta("样例库", "六类可识别缺陷")
            st.markdown(_sample_grid(), unsafe_allow_html=True)
            return

        height, width = image.shape[:2]
        meta_column, view_column = st.columns([2.4, 1.1], vertical_alignment="center")
        with meta_column:
            canvas_meta(filename or "", f"{width}×{height} px")
        with view_column:
            view = None
            if run is not None:
                view = st.segmented_control(
                    "视图",
                    VIEWS,
                    default="标注结果",
                    label_visibility="collapsed",
                    key="canvas_view",
                )
            else:
                st.markdown('<p class="sv-await">待检测</p>', unsafe_allow_html=True)

        if run is None:
            _stage(raw, filename or "待检测图片")
            st.caption("图片已载入。点右侧「开始检测」，结果会替换这张画布。")
            return

        images = {
            "标注结果": encode_png_bytes(run["annotated"]),
            "原图": raw,
            "九宫格": encode_png_bytes(run["grid"]),
        }
        _stage(images[view or "标注结果"], view or "标注结果")


# ---------------------------------------------------------------- 结果带


def render_ribbon(run: dict) -> None:
    """指标条：一次检测的四个事实，排成一条而不是四张浮卡。"""
    summary = run["summary"]
    section(
        "检测概览",
        f"{run['filename']} · 阈值 {run['conf']:.2f} · 单张耗时 {run['elapsed']:.0f} ms",
    )
    stat_ribbon(
        [
            ("检出缺陷", str(summary["total"]), "当前阈值下的检出数量"),
            ("涉及类别", str(len(summary["class_counts"])), "共 6 类可识别缺陷"),
            ("平均置信度", f"{summary['mean_confidence']:.3f}", "所有检出目标的均值"),
            ("单张耗时", f"{run['elapsed']:.0f} ms", f"{run['width']}×{run['height']} 输入"),
        ]
    )


def render_zone_band(run: dict) -> None:
    """区域分布与类别构成并排：左边给位置，右边给种类。"""
    analysis = run["analysis"]
    summary = run["summary"]

    zone_column, class_column = st.columns([1, 1.05], gap="large")

    with zone_column:
        st.subheader("九宫格区域分布")
        st.markdown(build_zone_tiles_html(analysis), unsafe_allow_html=True)
        if analysis.hotspot is None:
            st.info("本次未检出缺陷，无法判断缺陷集中区域。", icon=":material/info:")
        else:
            st.success(
                f"缺陷集中在 **{analysis.hotspot}**，该区域检出 {analysis.hotspot_count} 个。"
                "建议优先复核该区域对应的工艺环节。",
                icon=":material/location_on:",
            )

    with class_column:
        st.subheader("缺陷类别构成")
        if summary["total"] == 0:
            st.info(
                "未检出超过当前阈值的缺陷。注意：这不等于产品一定合格，仍需人工复核。",
                icon=":material/info:",
            )
        else:
            st.markdown(
                class_rows(summary["class_counts"], summary["total"]),
                unsafe_allow_html=True,
            )
            # 类别之外再看一眼把握：高 / 中 / 低各有多少个
            grades = {label: 0 for label in SEVERITY_LABELS.values()}
            for confidence in run["detections"].confidence:
                grades[SEVERITY_LABELS[severity_level(float(confidence))]] += 1
            st.markdown(
                '<p class="sv-grade-title">置信度分级（≥0.70 高 · 0.40–0.69 中 · 其余低）</p>'
                + grade_strip(grades),
                unsafe_allow_html=True,
            )


def render_detail_band(run: dict) -> None:
    """缺陷明细与导出：一行里放标题、两个导出按钮，下面整张表。"""
    detections = run["detections"]
    width, height = run["width"], run["height"]

    title_column, png_column, csv_column = st.columns([3, 1, 1], vertical_alignment="center")
    with title_column:
        st.markdown('<p class="sv-band-title">缺陷明细</p>', unsafe_allow_html=True)
    with png_column:
        st.download_button(
            "导出标注图",
            data=encode_png_bytes(run["annotated"]),
            file_name=f"{Path(run['filename']).stem}_annotated.png",
            mime="image/png",
            width="stretch",
        )
    with csv_column:
        csv_frame = pd.DataFrame(detections_to_records(detections, DEFAULT_CLASS_NAMES))
        st.download_button(
            "导出 CSV",
            data=csv_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{Path(run['filename']).stem}_detections.csv",
            mime="text/csv",
            width="stretch",
        )

    if len(detections) == 0:
        st.caption("当前阈值下没有检出目标，明细为空。")
        return

    rows = []
    for index in range(len(detections)):
        x1, y1, x2, y2 = (float(value) for value in detections.xyxy[index])
        confidence = float(detections.confidence[index])
        rows.append(
            {
                "序号": index + 1,
                "类别": DEFAULT_CLASS_NAMES[int(detections.class_id[index])],
                "区域": zone_of_box(x1, y1, x2, y2, width, height),
                "置信度": confidence,
                "分级": SEVERITY_LABELS[severity_level(confidence)],
                "框位置": f"{x1:.0f},{y1:.0f} → {x2:.0f},{y2:.0f}",
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=min(360, max(120, 40 + 35 * len(table))),
        column_config={
            "序号": st.column_config.NumberColumn("序号", width=56),
            "类别": st.column_config.TextColumn("类别", width=110),
            "区域": st.column_config.TextColumn("区域", width=92),
            "置信度": st.column_config.ProgressColumn(
                "置信度", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "分级": st.column_config.TextColumn("分级", width=92),
            "框位置": st.column_config.TextColumn("框位置 (x1,y1 → x2,y2)", width=250),
        },
    )


# ---------------------------------------------------------------- 批量模式


def _batch_grid(entries: list[tuple[str, str, bool]]) -> str:
    """批量缩略图网格：`(data_uri, 说明, 是否失败)` 列表。"""
    cells = ""
    for uri, caption, failed in entries:
        css = ' class="is-fail"' if failed else ""
        cells += (
            f'<figure><img src="{uri}" alt="{caption}">'
            f"<figcaption{css}>{caption}</figcaption></figure>"
        )
    return f'<div class="sv-samples">{cells}</div>'


def _batch_sources(state: dict) -> list[tuple[str, bytes]]:
    """本批要处理的图片：上传的文件优先，其次是用样例凑一批。"""
    uploaded = state.get("uploaded")
    if uploaded:
        return [(item.name, item.getvalue()) for item in uploaded][:MAX_BATCH_FILES]
    if state.get("use_samples"):
        pairs: list[tuple[str, bytes]] = []
        for filename, _ in CLASS_SAMPLES:
            path = PROJECT_ROOT / "samples" / filename
            if path.exists():
                pairs.append((path.name, path.read_bytes()))
        return pairs
    return []


def render_batch_canvas(
    files: list[tuple[str, bytes]],
    result,
    conf_threshold: float,
) -> None:
    """批量画布：未检测时看原图缩略图，检测后看标注缩略图与每张的缺陷数。"""
    with st.container(key="canvas"):
        if result is None:
            canvas_meta(f"已选 {len(files)} 张", f"阈值 {conf_threshold:.2f}")
            entries = [(_data_uri(raw), name, False) for name, raw in files[:6]]
            st.markdown(_batch_grid(entries), unsafe_allow_html=True)
            if len(files) > 6:
                st.caption(f"共 {len(files)} 张，这里只预览前 6 张；全部结果见下方明细。")
            st.caption("点右侧「开始批量检测」，系统逐张推理后在这里看到标注结果。")
            return

        canvas_meta(f"批量 · {len(result.items)} 张", f"阈值 {conf_threshold:.2f}", f"用时 {result.elapsed_text}")
        entries = [
            (f'data:image/png;base64,{base64.b64encode(encode_png_bytes(item.annotated)).decode()}',
             f"{item.name} · {item.count} 个",
             False)
            for item in result.items[:6]
            if item.ok and item.annotated is not None
        ]
        st.markdown(_batch_grid(entries), unsafe_allow_html=True)
        if len(result.items) > 6:
            st.caption(f"共 {len(result.items)} 张，这里只预览前 6 张。")
        if result.failed:
            st.caption(f"另有 {result.failed} 张失败，失败原因列在下方「异常与口径」。")


def render_batch_bands(result) -> None:
    """批量结果带：任务概览 → 整批类别构成 | 异常与口径 → 明细与 CSV。"""
    section(
        "批次概览",
        f"{len(result.items)} 张 · 阈值 {result.conf_threshold:.2f} · 用时 {result.elapsed_text}",
    )
    stat_ribbon(
        [
            ("已选图片", str(len(result.items)), "本批提交的张数"),
            ("已完成", str(result.completed), "含未检出"),
            ("累计缺陷", str(result.total_defects), "全批检出合计"),
            ("失败", str(result.failed), "读取或推理失败"),
        ]
    )

    class_column, note_column = st.columns([1, 1.05], gap="large")

    with class_column:
        st.subheader("整批类别构成")
        counts = result.class_counts
        if counts:
            st.markdown(
                class_rows(counts, result.total_defects), unsafe_allow_html=True
            )
        else:
            st.info(
                "本批没有检出超过阈值的缺陷。未检出不等于合格，仍需人工复核。",
                icon=":material/info:",
            )

    with note_column:
        st.subheader("异常与口径")
        failed = [item for item in result.items if not item.ok]
        # 失败清单与口径说明合并为一段，避免两条小字挤在一起
        lines = [f"· {item.name}：{item.error}" for item in failed[:5]]
        if len(failed) > 5:
            lines.append(f"· 另有 {len(failed) - 5} 张失败，见下方明细。")
        lines.append(
            f"逐张推理而非并行；单批最多 {MAX_BATCH_FILES} 张、单张不超过 20 MB；"
            "任一图片失败只记录原因，不影响其余图片。"
        )
        st.caption("\n".join(lines))

    title_column, csv_column = st.columns([3, 1], vertical_alignment="center")
    with title_column:
        st.markdown('<p class="sv-band-title">批量明细</p>', unsafe_allow_html=True)
    with csv_column:
        st.download_button(
            "导出 CSV",
            data=batch_csv(result),
            file_name=f"batch_{len(result.items)}张_{result.conf_threshold:.2f}.csv",
            mime="text/csv",
            width="stretch",
        )

    frame = batch_frame(result)
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        height=min(420, max(140, 38 * len(frame) + 40)),
        column_config={
            "图片名称": st.column_config.TextColumn("图片名称", width=200),
            "缺陷类型": st.column_config.TextColumn("缺陷类型", width=120),
            "置信度": st.column_config.NumberColumn("置信度", format="%.3f", width=90),
            "检测状态": st.column_config.TextColumn("检测状态", width=260),
        },
    )
    rule_note(
        "批量模式采用逐张推理，不代表多张图片并行处理；"
        "某张图片失败只记录原因并继续处理其余图片。"
        "「未检出」只表示当前阈值下没有检出目标，不代表产品合格。"
    )


# ---------------------------------------------------------------- 页脚


def render_footer() -> None:
    """页脚：数据来源在左，边界声明在右。"""
    footer_note(
        "NEU-DET 钢材表面缺陷数据集（清理修正版）· YOLOv8n · Supervision 0.32",
        "验证原型 · 检测结果用于辅助人工复检，「未检出缺陷」不等于产品合格",
    )


# ---------------------------------------------------------------- 主流程


def execute_detection(image: np.ndarray, conf_threshold: float) -> dict:
    """跑一次完整检测，返回画布与结果带需要的全部数据。"""
    with st.spinner("正在加载模型…"):
        model = get_model(str(WEIGHTS_PATH))

    with st.spinner("正在检测缺陷…"):
        start = time.perf_counter()
        detections = detect_image(
            model,
            image,
            conf_threshold=conf_threshold,
            class_names=DEFAULT_CLASS_NAMES,
        )
        elapsed = (time.perf_counter() - start) * 1000

    height, width = image.shape[:2]
    # 200 px 的样片用小一号标签并省掉置信度，否则文字会盖掉缺陷本身；
    # 置信度在指标条与明细表里都有，画布上不重复
    small = min(height, width) < 320
    return {
        "detections": detections,
        "annotated": annotate_image(
            image,
            detections,
            DEFAULT_CLASS_NAMES,
            text_scale=0.34 if small else 0.5,
            with_confidence=not small,
        ),
        "grid": draw_zone_grid(image),
        "summary": summarize_detections(detections, DEFAULT_CLASS_NAMES),
        "analysis": analyze_zones(detections, width=width, height=height),
        "elapsed": elapsed,
        "width": width,
        "height": height,
    }


def run_single_mode(canvas_column, state: dict) -> None:
    """单张模式：选一张图，出标注、九宫格与明细。"""
    uploaded = state["uploaded"]
    conf_threshold = state["conf_threshold"]
    start_clicked = state["start_clicked"]

    # 图片优先级：上传的文件 > 样例库
    payload: tuple[bytes, str] | None = None
    if uploaded is not None:
        payload = (uploaded.getvalue(), uploaded.name)
    elif state["sample_name"]:
        sample_path = PROJECT_ROOT / "samples" / SAMPLE_BY_NAME[state["sample_name"]]
        if sample_path.exists():
            payload = (sample_path.read_bytes(), sample_path.name)

    image: np.ndarray | None = None
    error: str | None = None
    if payload is not None:
        try:
            validate_upload(payload[1], len(payload[0]))
            image = decode_image_bytes(payload[0])
        except ValueError as exc:
            error = str(exc)

    # 结果按「文件 + 阈值」缓存：切换视图不会丢结果，换图或改阈值则作废
    run_key = f"{payload[1]}|{conf_threshold:.2f}" if payload else ""
    run = st.session_state.get("run")
    if run is not None and run.get("key") != run_key:
        run = None

    autorun = st.query_params.get("autorun") in ("1", "true", "yes")
    if image is not None and run is None and (start_clicked or autorun):
        if not WEIGHTS_PATH.exists():
            error = "未找到模型权重文件 weights/best.pt，请先运行 `python train.py` 完成训练。"
        else:
            try:
                run = execute_detection(image, conf_threshold)
                run.update(key=run_key, filename=payload[1], conf=conf_threshold)
                st.session_state["run"] = run
            except (ModelNotFoundError, DetectionError, ValueError) as exc:
                error = str(exc)

    with canvas_column:
        render_canvas(
            payload[1] if payload else None,
            payload[0] if payload else None,
            image,
            run,
            conf_threshold,
            error,
        )

    if error and image is not None:
        st.error(error, icon=":material/error:")

    if run is not None:
        render_ribbon(run)
        render_zone_band(run)
        render_detail_band(run)
        rule_note(
            f"区域计数合计 {run['analysis'].total} 个，参与统计的检测目标 "
            f"{run['analysis'].assigned_count} 个，未归属 "
            f"{run['analysis'].unassigned_count} 个。"
            "边界上的目标按从上到下、从左到右的顺序唯一归属，"
            "因此九个区域计数之和恒等于参与统计的检测目标数。"
        )


def run_batch_mode(canvas_column, state: dict) -> None:
    """批量模式：逐张推理、出汇总与 CSV；单张失败不影响整批。"""
    conf_threshold = state["conf_threshold"]
    files = _batch_sources(state)

    if not files:
        with canvas_column:
            with st.container(key="canvas"):
                canvas_meta("批量检测", "未选择图片")
                entries = [
                    (_data_uri((PROJECT_ROOT / "samples" / name).read_bytes()), name, False)
                    for name, _ in CLASS_SAMPLES
                    if (PROJECT_ROOT / "samples" / name).exists()
                ]
                if entries:
                    st.markdown(_batch_grid(entries), unsafe_allow_html=True)
                st.caption(
                    "上传多张图片，或打开右侧「用六类样例组成一批」，"
                    "系统会逐张推理并给出汇总与明细。"
                )
        return

    # 批次结果按「文件名序列 + 阈值」缓存，切换模式或改窗口不会重跑
    batch_key = "|".join(name for name, _ in files) + f"|{conf_threshold:.2f}"
    cached = st.session_state.get("batch")
    result = cached["result"] if cached and cached.get("key") == batch_key else None

    autorun = st.query_params.get("autorun") in ("1", "true", "yes")
    error: str | None = None

    with canvas_column:
        if result is None and (state["start_clicked"] or autorun):
            if not WEIGHTS_PATH.exists():
                error = "未找到模型权重文件 weights/best.pt，请先运行 `python train.py` 完成训练。"
            else:
                try:
                    model = get_model(str(WEIGHTS_PATH))
                    bar = st.progress(0.0, text=f"准备检测 {len(files)} 张…")

                    def _report(done: int, total: int, name: str) -> None:
                        if total <= 0:
                            return
                        shown = min(done + 1, total)
                        bar.progress(
                            min(done / total, 1.0),
                            text=f"正在检测 {shown}/{total}：{name}",
                        )

                    result = run_batch(
                        model,
                        files,
                        conf_threshold,
                        DEFAULT_CLASS_NAMES,
                        progress=_report,
                    )
                    bar.progress(1.0, text="检测完成")
                    bar.empty()
                    st.session_state["batch"] = {"key": batch_key, "result": result}
                except (ModelNotFoundError, DetectionError, ValueError) as exc:
                    error = str(exc)

        render_batch_canvas(files, result, conf_threshold)

    if error:
        st.error(error, icon=":material/error:")

    if result is not None:
        render_batch_bands(result)


def main() -> None:
    render_topbar()

    canvas_column, rail_column = st.columns([1.72, 1], gap="large")
    with rail_column:
        state = render_rail()

    if state["mode"] == MODES[1]:
        run_batch_mode(canvas_column, state)
    else:
        run_single_mode(canvas_column, state)

    render_footer()


if __name__ == "__main__":
    main()
