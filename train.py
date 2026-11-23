"""钢材表面缺陷检测模型训练脚本。

使用 Ultralytics YOLO 在 NEU-DET 数据集上训练缺陷检测模型，训练完成后
把 best.pt 复制到 weights/ 目录，供 Web 页面和 predict.py 使用。

用法：
    python train.py                          # 使用默认参数
    python train.py --epochs 50 --imgsz 640  # 自定义参数
    python train.py --model yolov8s.yaml     # 从配置文件开始训练
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "data" / "dataset" / "data.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "train"
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练钢材表面缺陷检测模型")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="起始模型，默认 yolov8n.pt（轻量，适合 CPU 训练）",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_YAML),
        help="数据集配置文件路径",
    )
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数，默认 30")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸，默认 640")
    parser.add_argument("--batch", type=int, default=8, help="批次大小，默认 8")
    parser.add_argument("--device", default="cpu", help="训练设备，默认 cpu")
    parser.add_argument("--workers", type=int, default=0, help="数据加载线程数，默认 0")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="训练输出目录",
    )
    parser.add_argument(
        "--name",
        default="steel_defect_yolov8n",
        help="本次训练的实验名称",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(
            f"未找到数据集配置：{data_yaml}\n"
            "请先运行 python prepare_dataset.py 整理数据集。"
        )

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=args.output,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        plots=True,
        verbose=True,
    )

    best_weights = Path(args.output) / args.name / "weights" / "best.pt"
    if not best_weights.exists():
        raise SystemExit(f"训练结束但未找到权重文件：{best_weights}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    target = WEIGHTS_DIR / "best.pt"
    shutil.copy2(best_weights, target)

    print("=" * 60)
    print("训练完成")
    print(f"  权重文件：{target}")
    print(f"  训练结果：{Path(args.output) / args.name}")
    print("  运行评估：python evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
