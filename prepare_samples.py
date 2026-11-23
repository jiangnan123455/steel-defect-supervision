"""导出演示用样例图片与标注结果图。

从测试集中挑选几张有代表性的图片，复制到 samples/ 目录，供 Web 页面
的"示例图片"入口和答辩演示使用。

用法：
    python prepare_samples.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
SAMPLES_DIR = PROJECT_ROOT / "samples"

# 每种缺陷类别各挑一张，作为演示素材
PICK_PER_CLASS = 1


def main() -> None:
    test_images = DATASET_DIR / "images" / "test"
    test_labels = DATASET_DIR / "labels" / "test"

    if not test_images.exists():
        raise SystemExit(
            f"未找到测试集目录：{test_images}\n请先运行 python prepare_dataset.py"
        )

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    picked: dict[int, str] = {}
    for label_path in sorted(test_labels.glob("*.txt")):
        class_ids = [
            int(line.split()[0])
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not class_ids:
            continue
        primary = class_ids[0]
        if picked.get(primary, 0) >= PICK_PER_CLASS:
            continue

        image_path = test_images / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue

        shutil.copy2(image_path, SAMPLES_DIR / image_path.name)
        picked[primary] = picked.get(primary, 0) + 1

    print(f"已导出 {len(picked)} 张示例图片到 {SAMPLES_DIR}")
    for class_id in sorted(picked):
        print(f"  class {class_id}")


if __name__ == "__main__":
    main()
