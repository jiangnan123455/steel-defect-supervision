---
license: other
license_name: see-license-file-upstream-unstated
license_link: LICENSE
task_categories:
- object-detection
language:
- en
tags:
- neu-det
- steel-surface-defect
- hot-rolled-steel
- industrial-inspection
- object-detection
- pascal-voc
- yolo
size_categories:
- 1K<n<10K
---

# NEU-DET-corrected

[![DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F10536-blue)](https://doi.org/10.57967/hf/10536)

**Hot-rolled steel strip surface defect detection — cleaned version of the NEU Surface Defect Database.**
**热轧带钢表面缺陷检测 —— NEU 表面缺陷数据库的清理修正版。**

[English](#english) · [中文](#中文)

---

## English

### What is this?

A **cleaned, complete and consistently packaged** version of the
**NEU Surface Defect Database (NEU-DET)** from Northeastern University
(Song Kechen & Yan Yunhui).

The upstream release ships as a flat `IMAGES/` + `ANNOTATIONS/` (Pascal VOC) pair with
no official split and a few annotation defects. This release keeps every image pixel
**unchanged**, removes a small number of duplicates, and packages the data in both
**Pascal VOC** and **YOLO** format with a reproducible train/val/test split.

> ⚠️ **Upstream license: none stated.** The original authors have never declared a license
> for NEU-DET. See [License & attribution](#license--attribution) — verify before commercial use.

### Corrections vs. the official release

| # | Correction | Detail |
|---|---|---|
| 1 | **Removed 1 exact-duplicate image** | `patches_105.jpg` is pixel-identical to `patches_101.jpg` (the same image annotated twice, with slightly different boxes). |
| 2 | **Removed 2 near-duplicate images** | `scratches_247.jpg` is a 90°-rotated copy of `scratches_232.jpg`; `pitted_surface_292.jpg` is a 2-px-shifted copy of `pitted_surface_171.jpg`. Both show a sharp, isolated minimum in the alignment profile (random same-class pairs never drop below MAE ≈ 13). |
| 3 | **Removed 3 duplicate boxes** | `crazing_120`, `inclusion_62`, `patches_198` each contained one bounding box listed twice verbatim. |
| 4 | **Added a reproducible split** | Upstream has no split. This release provides a seeded, class-stratified 80/10/10 split. |
| 5 | **Dual format** | Upstream is Pascal VOC only. This release ships **VOC + YOLO** from a single verified conversion. |
| 6 | **Packaging** | `classes.txt`, `data.yaml`, `ImageSets/` added. |

**Image pixels are never modified.** All 1,797 images are byte-for-byte identical to the upstream files.

**Scope — structural clean-up only.** This is **not** a re-annotation. We remove duplicates and
re-package the data; we do **not** re-verify individual boxes (missed defects, wrong class
labels, box tightness) — that requires domain expertise and is left untouched.

**Structural checks performed** on the released 1,797-image set:

| Check | Method | Result |
|---|---|---|
| Exact duplicate images | byte-wise MD5 | **1** found → removed |
| Rotation / mirror duplicates | 8 dihedral transforms, perceptual hash | **1** found → removed |
| Shifted duplicates | aligned pixel MAE + sharp-minimum test | **1** found → removed |
| Near-duplicate images | 32×32/64×64 MAE + full aligned scan | **0** remaining |
| Duplicate boxes within an image | verbatim box comparison | **3** found → removed |
| Empty / invalid labels | 0-box files, out-of-range, zero-area boxes | **0** |
| Image ↔ label ↔ XML pairing | bidirectional | **0** orphans |
| Class-name consistency | unique label set | 6 classes, no anomalies |
| Split leakage | train/val/test intersection | **0** |

### Dataset at a glance

| Property | Value |
|---|---|
| Images | **1,797** (200 × 200, grayscale) |
| Classes | **6** — `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches` |
| Bounding boxes | **4,177** (≈ 2.32 per image) |
| Formats | Pascal VOC XML · YOLO TXT |
| Split | train 1,437 / val 180 / test 180 (seeded, class-stratified) |

**Boxes per class:** `inclusion` 1,010 · `patches` 875 · `crazing` 688 · `rolled-in_scale` 628 ·
`scratches` 546 · `pitted_surface` 430

### Structure

```
NEU-DET-corrected/
├── images/{train,val,test}/   # 200x200 grayscale JPG
├── labels/{train,val,test}/   # YOLO: cls cx cy w h (normalized)
├── annotations/               # Pascal VOC XML (flat, cleaned)
├── ImageSets/{train,val,test}.txt
├── classes.txt
├── data.yaml                  # YOLO dataset config
├── LICENSE
└── README.md
```

`annotations/*.xml` and `labels/**/*.txt` are two representations of the **same** cleaned
annotations and are verified to agree.

### Citation

**1. The original dataset — please always cite this.**

```bibtex
@article{he2020neudet,
  title   = {An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features},
  author  = {He, Yu and Song, Kechen and Meng, Qinggang and Yan, Yunhui},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  volume  = {69}, number = {4}, pages = {1493--1504}, year = {2020},
  doi     = {10.1109/TIM.2019.2915404}
}
```

```bibtex
@article{song2013neudet,
  title   = {A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects},
  author  = {Song, Kechen and Yan, Yunhui},
  journal = {Applied Surface Science},
  volume  = {285}, pages = {858--864}, year = {2013},
  doi     = {10.1016/j.apsusc.2013.08.123}
}
```

**2. This corrected release — please cite it as well.** It is not identical to the official
release: 3 duplicate images (1 exact, 2 near-duplicates under rotation / shift) and 3 duplicate
boxes were removed, a reproducible split was added, and the data was re-packaged, so a citation
to the papers alone does not describe this version.

```bibtex
@misc{neudet_corrected,
  author    = {KeenForgeAI},
  title     = {NEU-DET-corrected: a cleaned release of the NEU Surface Defect Database},
  year      = {2026},
  version   = {1.0},
  publisher = {KeenForgeAI},
  url       = {https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected},
  note      = {Curated by Lu Gan and Sam Li. Derived from He et al. (2020),
               doi:10.1109/TIM.2019.2915404. Upstream licence unstated.}
}
```

**3. The annotation tool (optional).**

```bibtex
@software{keenforge,
  author    = {KeenForgeAI},
  title     = {KeenForge: a local-first, offline image annotation and model-training desktop tool},
  year      = {2026},
  publisher = {KeenForgeAI},
  url       = {https://github.com/KeenForgeAI/KeenForge},
  note      = {MIT licensed. Developed by Lu Gan and Sam Li.}
}
```

### License & attribution

**The upstream NEU-DET dataset states no license.** The original authors provide it for
research use via their [official page](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/)
(Google Drive / Baidu Netdisk) and only request a citation. The copyright status of the
original images and annotations is therefore **undetermined** — verify it before any
commercial use. See `LICENSE` in this repository.

Our modifications (de-duplication, re-packaging, split, documentation) are released openly.
Image pixels are unmodified and remain subject to the upstream terms.

### Provenance

- Source: official NEU-DET release (Google Drive), retrieved 2026-09-21.
- Upstream: 1,800 images + 1,800 Pascal VOC XML, no split.
- Verified: 1,797 images retained, byte-identical to upstream; 4,177 boxes after removing
  3 duplicate boxes and 3 duplicate images (1 exact, 2 near-duplicates).

---

## 中文

### 这是什么？

**NEU 表面缺陷数据库（NEU-DET）**（东北大学，宋克臣、颜云辉）的**清理、补全、规范化打包版**。

上游发布是扁平的 `IMAGES/` + `ANNOTATIONS/`（Pascal VOC），**无官方划分**，且存在少量标注缺陷。
本版**不修改任何图像像素**，仅删除少量重复项，并以 **Pascal VOC + YOLO 双格式**打包，
附带可复现的 train/val/test 划分。

> ⚠️ **上游未声明许可证。** 原作者从未为 NEU-DET 声明许可证。详见
> [许可证与署名](#许可证与署名)——商用前请自行确认。

### 相对官方版的修正

| # | 修正 | 说明 |
|---|---|---|
| 1 | **删除 1 张完全重复图** | `patches_105.jpg` 与 `patches_101.jpg` 像素完全相同（同一张图被标注两次，框略有差异）。 |
| 2 | **删除 2 张近似重复图** | `scratches_247.jpg` 是 `scratches_232.jpg` 旋转 90° 的副本；`pitted_surface_292.jpg` 是 `pitted_surface_171.jpg` 平移 2px 的副本。两者在对齐剖面中都呈**尖锐唯一极小值**（随机同类对从未低于 MAE ≈ 13）。 |
| 3 | **删除 3 个重复框** | `crazing_120`、`inclusion_62`、`patches_198` 各有 1 个边界框**逐字重复出现两次**。 |
| 4 | **补充可复现划分** | 上游无划分；本版提供固定随机种子、按类别分层的 80/10/10 划分。 |
| 5 | **双格式** | 上游仅有 Pascal VOC；本版同时提供 **VOC + YOLO**（由同一次转换生成并校验一致）。 |
| 6 | **规范化打包** | 新增 `classes.txt`、`data.yaml`、`ImageSets/`。 |

**图像像素从不修改**——1,797 张图与上游文件逐字节一致。

**本版范围——仅结构化清理。** 这是**结构性清理，不是重新标注**。我们只去重、重新打包；
**不**重新核对单个标注框（漏标、类别标错、框松紧）——那需要领域专业知识，一律保持原样。

**已执行的结构化检查**（针对发布的 1,797 张）：

| 检查 | 方法 | 结果 |
|---|---|---|
| 样本重复 · 精确 | 逐字节 MD5 | **1 组** → 已删 |
| 样本重复 · 旋转/镜像 | 8 种二面体变换感知哈希 | **1 组** → 已删 |
| 样本重复 · 平移 | 对齐像素 MAE + 尖锐极小值判定 | **1 组** → 已删 |
| 样本重复 · 近似 | 32×32/64×64 MAE + 全库对齐扫描 | **0 剩余** |
| 标签重复 · 图内 | 逐字比对同一图内的框 | **3 个** → 已删 |
| 空/非法标签 | 0 框文件、越界框、零面积框 | **0** |
| 图片↔标签↔XML 配对 | 双向 | **0 孤儿** |
| 类别名一致性 | 唯一类别名集合 | 6 类，无杂名 |
| 划分泄漏 | train/val/test 交集 | **0** |

### 数据集概览

| 属性 | 值 |
|---|---|
| 图片 | **1,797**（200 × 200 灰度） |
| 类别 | **6** —— `crazing`、`inclusion`、`patches`、`pitted_surface`、`rolled-in_scale`、`scratches` |
| 标注框 | **4,177**（约 2.32 框/图） |
| 格式 | Pascal VOC XML · YOLO TXT |
| 划分 | train 1,437 / val 180 / test 180（固定种子、类别分层） |

**各类框数**：`inclusion` 1,010 · `patches` 875 · `crazing` 688 · `rolled-in_scale` 628 ·
`scratches` 546 · `pitted_surface` 430

### 目录结构

```
NEU-DET-corrected/
├── images/{train,val,test}/   # 200x200 灰度 JPG
├── labels/{train,val,test}/   # YOLO：cls cx cy w h（归一化）
├── annotations/               # Pascal VOC XML（扁平，已清理）
├── ImageSets/{train,val,test}.txt
├── classes.txt
├── data.yaml                  # YOLO 配置
├── LICENSE
└── README.md
```

`annotations/*.xml` 与 `labels/**/*.txt` 是**同一份**清理后标注的两种表示，已校验一致。

### 引用

使用本数据集请**同时引用**原始工作与本修正版：

**1. 原始数据集（请务必引用）**

```bibtex
@article{he2020neudet,
  title   = {An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features},
  author  = {He, Yu and Song, Kechen and Meng, Qinggang and Yan, Yunhui},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  volume  = {69}, number = {4}, pages = {1493--1504}, year = {2020},
  doi     = {10.1109/TIM.2019.2915404}
}
```

```bibtex
@article{song2013neudet,
  title   = {A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects},
  author  = {Song, Kechen and Yan, Yunhui},
  journal = {Applied Surface Science},
  volume  = {285}, pages = {858--864}, year = {2013},
  doi     = {10.1016/j.apsusc.2013.08.123}
}
```

**2. 本修正版（请一并引用）** —— 本版**与官方发布并不相同**：删除了 1 张重复图与 3 个重复框，
补充了可复现划分，并重新打包，因此**只引用论文无法描述本版本**。

```bibtex
@misc{neudet_corrected,
  author    = {KeenForgeAI},
  title     = {NEU-DET-corrected: a cleaned release of the NEU Surface Defect Database},
  year      = {2026},
  version   = {1.0},
  publisher = {KeenForgeAI},
  url       = {https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected},
  note      = {Curated by Lu Gan and Sam Li. Derived from He et al. (2020),
               doi:10.1109/TIM.2019.2915404. Upstream licence unstated.}
}
```

**3. 标注工具（可选）**

```bibtex
@software{keenforge,
  author    = {KeenForgeAI},
  title     = {KeenForge: a local-first, offline image annotation and model-training desktop tool},
  year      = {2026},
  publisher = {KeenForgeAI},
  url       = {https://github.com/KeenForgeAI/KeenForge},
  note      = {MIT licensed. Developed by Lu Gan and Sam Li.}
}
```

### 许可证与署名

**上游 NEU-DET 未声明许可证。** 原作者通过
[官方主页](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/)（Google Drive / 百度网盘）
以研究用途提供，仅要求引用。因此原始图像与标注的**版权状态未定**——**商用前请自行确认**。
详见本仓库 `LICENSE`。

我们对数据集的修改部分（去重、重新打包、划分、文档）开源发布。图像像素未修改，
仍受上游条款约束。

### 来源

- 来源：官方 NEU-DET 发布（Google Drive），获取于 2026-09-21
- 上游：1,800 图 + 1,800 Pascal VOC XML，无划分
- 已校验：保留 1,797 图，与上游逐字节一致；删除 3 个重复框与 3 张重复图（1 张完全重复 + 2 张近似重复）后共 4,177 框
