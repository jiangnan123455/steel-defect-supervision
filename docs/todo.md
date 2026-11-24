# 待办清单（2026-11-24 收尾时状态）

> 本文件记录 11 月 24 日凌晨收尾时的项目状态与次日待办。
> 明天继续时先读这一份，再看 `docs/experiment.md`。

## 一、当前状态

### 已完成并验证

| 项 | 状态 |
|---|---|
| Streamlit Web 页面（方案书 P0 全部 8 项） | 完成，AppTest 零异常 |
| Supervision 标注（BoxAnnotator + LabelAnnotator） | 完成，中文类别名正常显示 |
| 3×3 九宫格区域分析（P1 创新点） | 完成，`sv.PolygonZone` + 中心点锚点 |
| NEU-DET 数据集 | 1797 张，1257/270/270 分层划分（seed=42） |
| 中文字体标签渲染器 | 完成，12 个测试覆盖 |
| 训练 / 评估 / 命令行 / 环境自检 / 演示脚本 | 完成并跑通 |
| 单元测试 | **54 个全绿** |
| 文档 | README + 3 份 docs + 数据许可说明 |
| Git | 已初始化并提交一次 |

### 训练进度（未完成）

训练在收尾时仍在后台运行：

```text
任务：python train.py --model weights/yolov8n.pt --epochs 40 --imgsz 416 --batch 16 --device cpu
输出：runs/train/steel_defect_yolov8n/
```

| 项 | 值 |
|---|---|
| 收尾时已训练轮数 | 约 10 / 40 |
| 单轮耗时 | 约 5.3 分钟（CPU，i7-12650H） |
| 剩余时间 | 约 2.5 小时 |
| 验证集最高 mAP50 | 0.562（第 8 轮） |
| 当前可用权重 | `weights/best.pt`（第 10 轮的中期权重） |

**当前 `weights/best.pt` 已经可用**，Web 页面明天可以直接跑起来演示。
中期评估已实测并写入 `docs/experiment.md`：

- Precision 0.736，Recall 0.486，mAP@0.5 0.481，mAP@0.5:0.95 0.242
- 测试集 270 张检出 595 个缺陷，区域计数合计 595（不变式成立）
- 单图耗时 303.5 ms（CPU）

## 二、明天的待办

### 优先级 P0

1. **等训练跑完，确认最终指标** —— ✅ 已恢复训练（`train.py --resume`），
   断点续训到 40 轮。**完成后**：
   ```powershell
   python evaluate.py --weights weights/best.pt --conf 0.25 --imgsz 416
   ```
   然后把 `docs/experiment.md` 与 `docs/defense_script.md` 里的
   `[最终值待更新]` 数字替换为最终值。

2. **功能测试复验** —— ✅ 已完成，10 项全部通过（见 experiment.md 第 5 节）。

3. **现场演示彩排** —— 待做。按方案书第 13 章流程：
   ```powershell
   python check_environment.py
   streamlit run app.py
   ```
   话术稿已写好：`docs/defense_script.md`。

### 优先级 P1

4. **人工核验区域分析指标** —— ✅ 已完成。用 NEU-DET 测试集人工标注为
   真值（`src/zone_eval.py`），实测：区域计数总绝对误差 503、单图平均
   1.86、热点命中率 0.444。结果与解读已写入 `docs/experiment.md`。

5. **答辩 PPT 与话术** —— ✅ 话术稿已完成（`docs/defense_script.md`，
   含七分钟结构和预答辩问答）。PPT 本体待做。

6. **GitHub 仓库整理** —— 代码已提交 3 次。剩余：演示截图/GIF、
   远程仓库推送。

### 优先级 P2

7. 批量检测、CSV 导出、检测历史记录（方案书 P2，可不做）

## 三、需要留意的坑

| 问题 | 说明 |
|---|---|
| 上游 NEU-DET 未声明许可 | 数据集不随仓库分发，README 已说明获取方式 |
| 416 分辨率漏检细长缺陷 | `crazing_101.jpg` 在 conf=0.25 下未检出，其他 5 张均正常检出；有 GPU 建议用 640 重训 |
| Windows 中文路径 | `cv2.imwrite` 会静默失败，一律用 `cv2.imencode(...).tofile(...)` |
| pip 网络 | 清华源没有 streamlit，用阿里云源 `https://mirrors.aliyun.com/pypi/simple/` |
| GitHub / HF 下载 | 直连被挡，需用 `curl -x http://127.0.0.1:7897` |
| supervision 的 FutureWarning | 库内部弃用提示，不影响结果，忽略即可 |

## 四、复现命令速查

```powershell
cd D:\新建文件夹\新建文件夹\新建文件夹\steel-defect-supervision-demo
.\.venv\Scripts\activate

python check_environment.py                    # 环境自检
python -m pytest                               # 54 个单元测试
streamlit run app.py                           # Web 页面
python predict.py --image samples/patches_109.jpg --conf 0.25
python demo_zone_analysis.py --image samples/patches_109.jpg --conf 0.25
python evaluate.py --weights weights/best.pt --conf 0.25 --imgsz 416
python train.py --model weights/yolov8n.pt --epochs 40 --imgsz 416 --batch 16 --device cpu
```
