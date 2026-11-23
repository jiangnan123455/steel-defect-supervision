"""在测试集上评估缺陷检测模型，并输出基线与改进方案的对照结果。

基线方案：YOLO 检测 + 基础目标框展示
改进方案：YOLO + Supervision 标注 + 九宫格区域统计

两个方案使用同一模型权重、同一测试集、同一置信度阈值，因此模型检测
指标（Precision / Recall / mAP）理论上应完全一致；改进方案的价值体现在
区域统计与可视化指标上。

用法：
    python evaluate.py
    python evaluate.py --conf 0.25 --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "data" / "dataset" / "data.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

from src.detection import DEFAULT_CLASS_NAMES  # noqa: E402
from src.zone_analysis import ZONE_NAMES, analyze_zones  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估缺陷检测模型并生成实验报告")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="模型权重路径")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="数据集配置")
    parser.add_argument("--split", default="test", help="评估使用的数据子集，默认 test")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    parser.add_argument("--device", default="cpu", help="推理设备，默认 cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸，默认 640")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"未找到模型权重：{weights}\n请先运行 python train.py 完成训练。"
        )

    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.to(args.device)

    metrics = model.val(
        data=args.data,
        split=args.split,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        project=str(PROJECT_ROOT / "runs" / "val"),
        name="steel_defect_eval",
        exist_ok=True,
        verbose=False,
    )

    # 区域统计：以测试集图片为单位，统计模型检出目标在九宫格中的分布
    zone_totals = {name: 0 for name in ZONE_NAMES}
    per_image_counts: list[int] = []
    elapsed_total = 0.0
    image_count = 0

    image_dir = Path(args.data).parent / "images" / args.split
    label_dir = Path(args.data).parent / "labels" / args.split

    from src.detection import detect_image

    for image_path in sorted(image_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_count += 1

        start = time.perf_counter()
        detections = detect_image(
            model,
            image,
            conf_threshold=args.conf,
            class_names=DEFAULT_CLASS_NAMES,
        )
        elapsed_total += time.perf_counter() - start

        height, width = image.shape[:2]
        analysis = analyze_zones(detections, width=width, height=height)
        for name, count in analysis.counts.items():
            zone_totals[name] += count
        per_image_counts.append(analysis.total)

    hotspot = max(zone_totals.items(), key=lambda item: item[1])[0] if sum(zone_totals.values()) else None

    report = {
        "experiment": "EXP-02",
        "model": weights.name,
        "split": args.split,
        "conf_threshold": args.conf,
        "images_evaluated": image_count,
        "detection_metrics": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
        "zone_analysis": {
            "zone_counts": zone_totals,
            "hotspot_zone": hotspot,
            "mean_defects_per_image": (
                float(np.mean(per_image_counts)) if per_image_counts else 0.0
            ),
        },
        "runtime": {
            "total_inference_seconds": round(elapsed_total, 3),
            "mean_ms_per_image": (
                round(elapsed_total / image_count * 1000, 2) if image_count else 0.0
            ),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 60)
    print("模型检测指标")
    for key, value in report["detection_metrics"].items():
        print(f"  {key}: {value:.4f}")
    print("区域分析")
    print(f"  九宫格缺陷合计: {sum(zone_totals.values())}")
    print(f"  缺陷集中区域: {hotspot}")
    print(f"  单图平均缺陷数: {report['zone_analysis']['mean_defects_per_image']:.2f}")
    print(f"  单图平均耗时: {report['runtime']['mean_ms_per_image']:.1f} ms")
    print(f"报告已保存：{report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
