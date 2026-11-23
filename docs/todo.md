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

1. **等训练跑完，确认最终指标**
   ```powershell
   cd D:\新建文件夹\新建文件夹\新建文件夹\steel-defect-supervision-demo
   .\.venv\Scripts\activate
   # 训练若已中断，用 --resume 续训：
   # python -c "from ultralytics import YOLO; YOLO('runs/train/steel_defect_yolov8n/weights/last.pt').train(resume=True)"
   ```
   训练脚本结束时会自动把 `best.pt` 复制到 `weights/`。若手动中断，需手动复制：
   ```powershell
   Copy-Item runs\steel_defect_yolov8n\weights\best.pt weights\best.pt
   ```

2. **在最终权重上重跑评估，更新文档**
   ```powershell
   python evaluate.py --weights weights/best.pt --conf 0.25 --imgsz 416
   ```
   然后把 `docs/experiment.md` 里 EXP-01 / EXP-02 两行的数值、
   区域分布表、推理性能表更新为最终值，并删掉"中期评估"字样。

3. **功能测试复验**（方案书第 11 章）
   重点人工确认 T01、T05、T09、T10：
   - 上传 JPG → 页面显示原图
   - 点击"开始检测" → 返回结果
   - 下载标注图 → 能正常打开
   - 拖动置信度阈值 → 检出数量随阈值变化

4. **现场演示彩排**（按方案书第 13 章流程）
   ```powershell
   python check_environment.py
   streamlit run app.py
   ```
   准备 2–3 张本地测试图片 + 一份录屏作为现场异常时的备份。

### 优先级 P1

5. **人工核验区域分析指标**
   方案书要求"区域计数绝对误差"和"热点区域判断准确率"必须用人工标注
   作为真值，不能只用模型自身输出。做法：随机抽 20 张测试图，人工数每个
   区域的缺陷数，与系统输出比对，填入 `docs/experiment.md`。

6. **答辩 PPT 与话术**
   按方案书 13.1 的七分钟结构准备，重点讲"Supervision 在检测结果之上
   增加空间分布分析"这一创新点，明确说明它不改变模型精度。

7. **GitHub 仓库整理**
   - `git add -A && git commit`
   - 权重文件按 `.gitignore` 排除，README 里写清获取方式
   - 补 `docs/` 下的演示截图或 GIF

### 优先级 P2

8. 批量检测、CSV 导出、检测历史记录（方案书 P2，可不做）

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
