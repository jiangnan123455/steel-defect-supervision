"""NEU-DET 数据集整理脚本。

把 data/neu-det 下的原始图片与 YOLO 标签按类别分层，重新划分为
训练集 70% / 验证集 15% / 测试集 15%，并生成 data/data.yaml。

划分使用固定随机种子，保证实验可复现；同一张图片的图片与标签始终
放在同一个子集中。

用法：
    python prepare_dataset.py                 # 使用默认参数重新划分
    python prepare_dataset.py --seed 42 --ratios 0.7 0.15 0.15
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "neu-det"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def collect_samples(images_dir: Path, labels_dir: Path) -> list[tuple[str, list[int]]]:
    """收集 (图片文件名, 类别集合) 列表。

    原始数据里图片与标签都放在 train/val/test 子目录下，因此按文件名
    在标签目录中递归查找对应的标注文件。
    """
    label_index = {path.stem: path for path in labels_dir.rglob("*.txt")}

    samples: list[tuple[str, list[int]]] = []
    for image_path in sorted(images_dir.rglob("*.jpg")):
        label_path = label_index.get(image_path.stem)
        if label_path is None:
            continue
        class_ids = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                class_ids.append(int(line.split()[0]))
        samples.append((image_path.name, class_ids))
    return samples


def stratified_split(
    samples: list[tuple[str, list[int]]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[str]]:
    """按类别分层划分数据集。

    以"图片的主类别"为分层依据，保证六个类别在三个子集中的比例接近。
    """
    rng = random.Random(seed)
    by_class: dict[int, list[str]] = defaultdict(list)
    for name, class_ids in samples:
        primary = class_ids[0] if class_ids else -1
        by_class[primary].append(name)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_names = ["train", "val", "test"]

    for class_id in sorted(by_class):
        names = sorted(by_class[class_id])
        rng.shuffle(names)
        total = len(names)
        train_end = round(total * ratios[0])
        val_end = train_end + round(total * ratios[1])
        splits["train"].extend(names[:train_end])
        splits["val"].extend(names[train_end:val_end])
        splits["test"].extend(names[val_end:])

    for split_name in split_names:
        splits[split_name].sort()
    return splits


def copy_files(splits: dict[str, list[str]], images_dir: Path, labels_dir: Path) -> None:
    """把划分结果复制到输出目录。"""
    import shutil

    image_index = {path.name: path for path in images_dir.rglob("*.jpg")}
    label_index = {path.stem: path for path in labels_dir.rglob("*.txt")}

    for split_name, names in splits.items():
        split_images = OUTPUT_DIR / "images" / split_name
        split_labels = OUTPUT_DIR / "labels" / split_name
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)

        for name in names:
            stem = Path(name).stem
            shutil.copy2(image_index[name], split_images / name)
            shutil.copy2(label_index[stem], split_labels / f"{stem}.txt")


def write_data_yaml(splits: dict[str, list[str]]) -> Path:
    """生成 YOLO 训练用的 data.yaml。"""
    lines = [
        "# 由 prepare_dataset.py 自动生成，请勿手工修改",
        f"path: {OUTPUT_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "nc: 6",
        "names:",
    ]
    for index, name in enumerate(CLASS_NAMES):
        lines.append(f"  {index}: {name}")

    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 NEU-DET 数据集并生成分层划分")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.15, 0.15),
        help="训练/验证/测试集比例，默认 0.7 0.15 0.15",
    )
    args = parser.parse_args()

    images_dir = RAW_DIR / "images"
    labels_dir = RAW_DIR / "labels"
    if not images_dir.exists():
        raise SystemExit(f"未找到原始数据目录：{images_dir}")

    samples = collect_samples(images_dir, labels_dir)
    if not samples:
        raise SystemExit(f"未在 {images_dir} 中找到可用的图片，请先下载 NEU-DET 数据集。")

    splits = stratified_split(samples, tuple(args.ratios), args.seed)
    copy_files(splits, images_dir, labels_dir)
    yaml_path = write_data_yaml(splits)

    print(f"共处理 {len(samples)} 张图片")
    for name, items in splits.items():
        print(f"  {name}: {len(items)}")
    print(f"数据配置已写入：{yaml_path}")


if __name__ == "__main__":
    main()
