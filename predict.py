"""单张图片缺陷检测命令行脚本。

在无 Web 环境下快速验证模型与标注效果，同时输出九宫格区域统计。

用法：
    python predict.py --image data/dataset/images/test/crazing_1.jpg
    python predict.py --image sample.jpg --conf 0.3 --output outputs/images
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.annotation import annotate_image
from src.detection import DEFAULT_CLASS_NAMES, DetectionError, detect_image, load_model
from src.utils import detections_to_records, summarize_detections
from src.zone_analysis import analyze_zones

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对单张钢材图片执行缺陷检测")
    parser.add_argument("--image", required=True, help="待检测图片路径")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    parser.add_argument("--device", default="cpu", help="推理设备，默认 cpu")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="结果图片输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"未找到图片：{image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"无法读取图片：{image_path}")

    try:
        model = load_model(args.weights, device=args.device)
    except DetectionError as error:
        raise SystemExit(str(error)) from error

    height, width = image.shape[:2]

    start = time.perf_counter()
    try:
        detections = detect_image(
            model,
            image,
            conf_threshold=args.conf,
            class_names=DEFAULT_CLASS_NAMES,
        )
    except DetectionError as error:
        raise SystemExit(str(error)) from error
    elapsed = (time.perf_counter() - start) * 1000

    annotated = annotate_image(image, detections, DEFAULT_CLASS_NAMES)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_annotated.jpg"
    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        raise SystemExit("结果图片编码失败")
    encoded.tofile(str(output_path))

    summary = summarize_detections(detections, DEFAULT_CLASS_NAMES)
    analysis = analyze_zones(detections, width=width, height=height)

    print("=" * 60)
    print(f"图片：{image_path.name}  尺寸：{width}×{height}")
    print(f"耗时：{elapsed:.0f} ms")
    print(f"检出缺陷：{summary['total']} 个，平均置信度 {summary['mean_confidence']:.3f}")
    if summary["class_counts"]:
        print("类别分布：")
        for name, count in summary["class_counts"].items():
            print(f"  {name}: {count}")
    else:
        print("类别分布：未检出超过阈值的缺陷（不等于产品合格，仍需人工复核）")

    print("九宫格区域分布：")
    for name in analysis.counts:
        print(f"  {name}: {analysis.counts[name]}")
    if analysis.hotspot:
        print(f"缺陷集中区域：{analysis.hotspot}（{analysis.hotspot_count} 个）")

    print("缺陷明细：")
    for record in detections_to_records(detections, DEFAULT_CLASS_NAMES):
        print(
            f"  #{record['序号']} {record['类别']} conf={record['置信度']:.3f} "
            f"box=({record['x1']:.0f},{record['y1']:.0f})-({record['x2']:.0f},{record['y2']:.0f})"
        )

    print(f"结果已保存：{output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
