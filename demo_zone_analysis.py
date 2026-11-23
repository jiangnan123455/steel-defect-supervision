"""九宫格区域分析演示脚本。

在命令行下演示"缺陷 → Supervision 标注 → 九宫格区域统计"的完整链路，
用于答辩讲解或生成演示截图。

用法：
    python demo_zone_analysis.py --image samples/crazing_101.jpg
    python demo_zone_analysis.py --image samples/scratches_101.jpg --conf 0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.annotation import annotate_image
from src.chinese_label import find_chinese_font
from src.detection import DEFAULT_CLASS_NAMES, detect_image, load_model
from src.utils import detections_to_records
from src.zone_analysis import ZONE_NAMES, analyze_zones, build_grid_polygons

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "images"

# 九宫格热力配色：低 → 高
HEAT_COLORS = [
    (232, 245, 238),
    (214, 239, 219),
    (253, 231, 200),
    (250, 196, 160),
    (238, 130, 100),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="九宫格区域分析演示")
    parser.add_argument("--image", required=True, help="待分析图片路径")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--device", default="cpu", help="推理设备")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    return parser.parse_args()


def heat_color(ratio: float) -> tuple[int, int, int]:
    """按区域缺陷占比取热力颜色。"""
    if ratio <= 0:
        return HEAT_COLORS[0]
    index = min(int(ratio * (len(HEAT_COLORS) - 1)) + 1, len(HEAT_COLORS) - 1)
    return HEAT_COLORS[index]


def draw_text(image: np.ndarray, text: str, position: tuple[int, int], size: int = 14) -> None:
    """在图上绘制文字，优先使用中文字体，无中文字体时降级为 OpenCV。"""
    font = find_chinese_font(size)
    if font is None:
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            size / 28,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        return

    from PIL import Image, ImageDraw

    canvas = Image.fromarray(image)
    drawer = ImageDraw.Draw(canvas)
    drawer.text(position, text, font=font, fill=(40, 40, 40))
    image[:] = np.array(canvas)[:, :, ::-1]


def draw_zone_heatmap(image: np.ndarray, analysis) -> np.ndarray:
    """在原图上绘制九宫格热力底色、分割线与区域统计。"""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    polygons = build_grid_polygons(width, height)

    max_count = max(analysis.counts.values()) if analysis.counts else 0

    overlay = canvas.copy()
    for index, polygon in enumerate(polygons):
        name = ZONE_NAMES[index]
        count = analysis.counts.get(name, 0)
        ratio = count / max_count if max_count else 0.0
        points = polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [points], heat_color(ratio))

    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    for index, polygon in enumerate(polygons):
        name = ZONE_NAMES[index]
        count = analysis.counts.get(name, 0)
        points = polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, (90, 90, 90), 1)

        min_xy = polygon.min(axis=0)
        max_xy = polygon.max(axis=0)
        cx, cy = int((min_xy[0] + max_xy[0]) / 2), int((min_xy[1] + max_xy[1]) / 2)
        draw_text(canvas, name, (cx - 26, cy - 20))
        draw_text(canvas, str(count), (cx - 8, cy + 4), size=20)

    return canvas


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"未找到图片：{image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"无法读取图片：{image_path}")

    model = load_model(args.weights, device=args.device)
    height, width = image.shape[:2]

    detections = detect_image(
        model, image, conf_threshold=args.conf, class_names=DEFAULT_CLASS_NAMES
    )
    annotated = annotate_image(image, detections, DEFAULT_CLASS_NAMES)
    analysis = analyze_zones(detections, width=width, height=height)
    heatmap = draw_zone_heatmap(image, analysis)

    # 把标注图与热力图横向拼接
    combined = np.hstack([annotated, heatmap])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_zone_demo.jpg"
    success, encoded = cv2.imencode(".jpg", combined)
    if not success:
        raise SystemExit("图片编码失败")
    encoded.tofile(str(output_path))

    print("=" * 60)
    print(f"图片：{image_path.name}（{width}×{height}）")
    print(f"检出缺陷：{len(detections)} 个")
    print("九宫格区域分布：")
    for name in ZONE_NAMES:
        bar = "#" * analysis.counts.get(name, 0)
        print(f"  {name}: {analysis.counts.get(name, 0):>2}  {bar}")
    if analysis.hotspot:
        print(f"缺陷集中区域：{analysis.hotspot}（{analysis.hotspot_count} 个）")
    print("缺陷明细：")
    for record in detections_to_records(detections, DEFAULT_CLASS_NAMES):
        print(
            f"  #{record['序号']} {record['类别']} conf={record['置信度']:.3f} "
            f"box=({record['x1']:.0f},{record['y1']:.0f})-({record['x2']:.0f},{record['y2']:.0f})"
        )
    print(f"演示图已保存：{output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
