# 数据集说明

## 数据来源

本项目使用 **NEU-DET 钢材表面缺陷数据集**（NEU Surface Defect Database），
由东北大学（Song Kechen & Yan Yunhui）发布，是钢材表面缺陷检测领域的
公开基准数据集。

实际下载使用的是 HuggingFace 上的清理修正版
[`KeenForgeAI/NEU-DET-corrected`](https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected)：

- 图片像素与官方发布版本逐字节一致，未做任何修改；
- 修正了 1 张完全重复图片、2 张近重复图片、3 处重复标注框；
- 同时提供 Pascal VOC 与 YOLO 两种标注格式；
- 附带固定种子的可复现划分。

> **许可提示**：上游 NEU-DET 官方发布**未声明任何许可协议**。该清理修正版
> 仅对其自身的修改部分采用 CC BY 4.0，图片版权归原作者所有。公开仓库中
> 不包含本目录的图片文件，README 中说明获取方式。使用前请自行核对使用
> 范围是否符合要求。

## 类别定义

六类钢材表面缺陷，训练时的 `class_id` 顺序如下：

| class_id | 英文类别 | 中文名称 |
|---:|---|---|
| 0 | crazing | 龟裂 |
| 1 | inclusion | 夹杂 |
| 2 | patches | 压痕斑块 |
| 3 | pitted_surface | 麻点 |
| 4 | rolled-in_scale | 氧化铁皮压入 |
| 5 | scratches | 划痕 |

## 数据规模

| 项 | 数量 |
|---|---:|
| 图片总数 | 1797 |
| 每张图片的缺陷实例 | 1 个（NEU-DET 的原始设定） |

## 目录结构

原始数据放在 `data/neu-det/`：

```text
data/neu-det/
├── images/
│   ├── train/   val/   test/
├── labels/
│   ├── train/   val/   test/
├── annotations/          # Pascal VOC XML 标注
├── classes.txt
├── data.yaml
├── LICENSE
└── SOURCE_README.md      # 上游数据说明
```

YOLO 标签每行格式为：

```text
class_id x_center y_center width height
```

坐标按图片宽高归一化到 `[0, 1]`。

## 数据划分

使用 `prepare_dataset.py` 按**类别分层**重新划分，固定随机种子 `42`，
保证实验可复现：

| 子集 | 图片数 | 占比 |
|---|---:|---:|
| 训练集 | 1257 | 70% |
| 验证集 | 270 | 15% |
| 测试集 | 270 | 15% |

划分后的数据放在 `data/dataset/`，其中 `data.yaml` 由脚本自动生成：

```text
data/dataset/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── data.yaml
```

重新划分：

```bash
python prepare_dataset.py --seed 42 --ratios 0.7 0.15 0.15
```

### 数据泄漏说明

清理修正版已移除 1 张完全重复图片与 2 张近重复图片，避免相似样本跨集合。
本项目在此基础上按类别分层随机划分，不做额外增强裁剪，因此同一张原图
不会同时出现在训练集和测试集中。

## 获取数据

数据不随 Git 仓库分发。获取方式：

```bash
# 需要安装 git-lfs
git lfs install
git clone https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected data/neu-det
```

或从 <https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected> 直接下载。
下载完成后运行 `python prepare_dataset.py` 生成本项目使用的划分。

## 引用

```text
L. Song, Y. Yan, et al. NEU Surface Defect Database.
```

论文：Song Kechen, Yan Yunhui. *An automatic defects detection for steel strip surface
based on convolutional neural network trained with adversarial occlusion sample
augmentation*. ICIP 2013.
