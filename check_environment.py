"""项目结构自检脚本。

启动 Web 应用前运行，快速确认依赖、数据集、模型权重是否就位，
避免现场演示时才发现环境问题。

用法：
    python check_environment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))


def main() -> None:
    for module_name in ["cv2", "numpy", "pandas", "plotly", "streamlit", "supervision", "ultralytics"]:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            check(f"依赖 {module_name}", True, version)
        except ImportError as error:
            check(f"依赖 {module_name}", False, str(error))

    weights = PROJECT_ROOT / "weights" / "best.pt"
    check(
        "模型权重 weights/best.pt",
        weights.exists(),
        f"{weights.stat().st_size / 1024 / 1024:.1f} MB" if weights.exists() else "缺失，请运行 python train.py",
    )

    data_yaml = PROJECT_ROOT / "data" / "dataset" / "data.yaml"
    check(
        "数据集配置 data/dataset/data.yaml",
        data_yaml.exists(),
        "就绪" if data_yaml.exists() else "缺失，请运行 python prepare_dataset.py",
    )

    for split in ["train", "val", "test"]:
        images = PROJECT_ROOT / "data" / "dataset" / "images" / split
        count = len(list(images.glob("*.jpg"))) if images.exists() else 0
        check(f"数据集子集 {split}", count > 0, f"{count} 张图片")

    samples = PROJECT_ROOT / "samples"
    sample_count = len(list(samples.glob("*.jpg"))) if samples.exists() else 0
    check("演示样例图片", sample_count > 0, f"{sample_count} 张")

    print("=" * 66)
    print("环境自检结果")
    print("=" * 66)
    failed = 0
    for name, ok, detail in CHECKS:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {name:<38} {detail}")
        if not ok:
            failed += 1
    print("=" * 66)

    if failed:
        print(f"存在 {failed} 项未通过，请先修复后再启动 Web 应用。")
        sys.exit(1)
    print("全部检查通过，可以运行 streamlit run app.py")


if __name__ == "__main__":
    main()
