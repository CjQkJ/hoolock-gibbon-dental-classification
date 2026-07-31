# 白眉长臂猿牙齿图像分类：论文复现代码

本仓库是面向论文审稿的独立代码包，覆盖数据清单校验、离线增强、按冻结的 SAM31 协议训练和评估 31 个模型、结果汇总，以及 ConvNeXt Grad-CAM。面向审稿人的代码不依赖原服务器路径，数据集、清单和模型权重位置均通过命令行参数传入。

英文说明见 [README.md](README.md)。

## 冻结协议

当前 main 分支以冻结协议 `sam31_e73b33b_v1` 为准，规范候选为 `e73b33b`（ConvNeXt-Base，`convnext_base.fb_in22k_ft_in1k`）。

- `configs/sam31_e73b33b.json`：规范协议定义。
- `configs/sam31_e73b33b_models.lock.json`：服务器端 31 个模型权重和运行例外的审计锁定文件。
- `configs/models_31.json`：可移植的 31 模型注册表，含相对权重路径和 SHA-256 哈希。
- `results/table_s4_results.csv`：论文 Table S4 使用的最终 31 模型指标。

## 论文实验口径

- 二分类标签：`H. hoolock` 与 `H. leuconedys group`；测试集中的 `H. tianxing` 并入后者。
- 清理后原始数据：75 个体、340 张图片。
- 训练原图：258 张、60 个体。
- 最终训练集：1984 张，包括 258 张原图、436 张博物馆风格化图像和 1290 张离线增强图像。
- 测试集：82 张原图、15 个体；个体不跨训练集和测试集。
- 在线增强：直接正方形缩放至各模型原生输入尺寸、水平翻转 `p=0.5`、5 度旋转 `p=0.5`、亮度抖动 `0.1`。冻结协议不使用 `RandomResizedCrop`。
- 优化：AdamW，初始学习率 `3e-4`，权重衰减 `0.01`，余弦退火，最多 200 epoch，早停 patience 30。
- SAM：非自适应 SAM，`rho=0.05`，参数精确恢复，对所有可训练梯度计算全局 L2 范数。
- 损失：按训练类别频数倒数加权的交叉熵。
- 选择指标：Macro-F1，因没有单独验证集而在测试集上监测。
- 输入大小：各模型使用 `configs/models_31.json` 记录的预训练分辨率。

逻辑 batch size 为 16，默认物理 micro-batch 为 16。模型注册表记录的审计例外如下：

- EfficientNet-B7：输入 600，物理 micro-batch 4，逻辑 batch 仍为 16。
- MobileNetV5-300M：冻结 backbone，物理 micro-batch 4，仅训练分类头。
- SwinV2-Large：两卡 DataParallel。

## 重要评估说明

原实验没有单独验证集。测试集同时用于逐 epoch 早停、最佳 checkpoint 选择和最终指标报告。因此结果是“测试集参与模型选择”的实验结果，可能比完全独立测试评估乐观。发布代码忠实保留这一流程，并在 `selection_split: test` 中显式记录。

测试推理为确定性流程：每张图一个视图、一次前向、无 TTA、无校准、无阈值调参、无集成，决策规则为 `argmax`。

## 文件结构

```text
configs/                      SAM31 协议、模型锁定文件和 31 模型注册表
metadata/dataset_manifest.csv   可移植的 2066 行图像清单
results/table_s4_results.csv   论文 Table S4 的最终 SAM31 31 模型结果
scripts/                      清单、离线增强、批量运行和汇总脚本
src/                          数据、变换、模型、SAM 训练、指标和 Grad-CAM
tests/                        计数、指标、协议元数据和变换测试
```

## 环境

原实验环境为 PyTorch `2.10.0+cu128`、CUDA `12.8`、cuDNN `9.10.2`、timm `1.0.25`，GPU 型号为 NVIDIA GeForce RTX 4090 24 GB。结果记录中 30 个模型使用单卡，SwinV2-Large 使用两卡 DataParallel。安装示例：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0
pip install -r requirements.txt
```

## 数据目录

代码预期如下结构；中间允许存在风格化来源子目录和个体目录：

```text
dataset_augmented/
  train/
    hoolock/**/*.jpg
    leuconedys/**/*.jpg
  test/
    hoolock/**/*.jpg
    leuconedys/**/*.jpg
```

校验发布数据并重建 manifest：

```bash
python scripts/build_manifest.py \
  --data-root /path/to/dataset_augmented \
  --output metadata/dataset_manifest.csv \
  --strict-paper-counts
```

严格校验应输出训练 `966/1018`、测试 `28/54`，训练/测试个体分别为 `60/15`。

## 离线增强

每张训练原图从 8 类颜色预设中无放回随机选择 5 类：`bright_sat`、`dark_con`、`warm`、`cool`、`hicon`、`soft`、`vivid` 和 `muted`。参数完整记录在 `scripts/offline_augmentation.py`。

```bash
python scripts/offline_augmentation.py \
  --source-root /path/to/original_train_test \
  --output-root /path/to/generated_dataset \
  --seed 20260430
```

历史正式数据生成时没有保存离线增强的随机数状态；发布脚本增加了显式 seed 以保证后续运行可复现。论文使用的确切增强文件由发布 manifest 固定。

436 张博物馆风格化图像是预先生成并随正式数据集管理的固定输入，不在训练时生成。当前项目中没有保存其原始风格化生成模型，因此本包复现的是以发布版 `dataset_augmented` 为输入的完整训练流程。

## 训练

先进行不下载权重的结构和数据检查：

```bash
python -m src.train \
  --model-key convnext_base \
  --data-root /path/to/dataset_augmented \
  --no-pretrained --dry-run
```

要严格复现锁定协议，请通过 `--checkpoint` 传入审计过的服务器权重，或在批量脚本中使用 `--checkpoint-root`：

```bash
python -m src.train \
  --model-key convnext_base \
  --data-root /path/to/dataset_augmented \
  --checkpoint /path/to/weights/18_convnext_base.fb_in22k_ft_in1k/model.safetensors
```

未传 checkpoint 时，timm 会在可用时尝试下载预训练权重。锁定权重不存储在本仓库中；`models_31.json` 记录其相对路径和 SHA-256，`sam31_e73b33b_models.lock.json` 记录服务器端审计绝对路径。

使用注册表里的模型专用 GPU 和 micro-batch 设置依次运行 31 个模型：

```bash
nohup python scripts/run_model_zoo.py \
  --data-root /path/to/dataset_augmented \
  --checkpoint-root /path/to/weights \
  --continue-on-error > model_zoo.log 2>&1 &
```

`--checkpoint-root` 可选；传入后，每个模型会以 `--checkpoint <root>/<weight_relative_path>` 启动，路径来自 `configs/models_31.json`。

## 结果

`results/table_s4_results.csv` 包含全部 31 个模型的最终 SAM31 结果。冻结运行中的最优模型为 ConvNeXt-Base：

- Accuracy：92.68%（76/82）
- Balanced accuracy：90.15%
- Macro-F1：91.55%
- KIZ011338：2/2 正确
- MCZ26474：16/16 正确

这些结果以测试集作为模型选择划分，论文中应明确描述为 test-guided optimization。

## Grad-CAM

论文选用 ConvNeXt-Base 进行形态解释。目标层为 `stages.3`，默认对真实类别求梯度，使用完整热力图覆盖，不设置 cutoff 或 mask。

```bash
python -m src.gradcam \
  --checkpoint /path/to/convnext_best_model.pth \
  --data-root /path/to/dataset_augmented \
  --split test \
  --source-type original \
  --target true \
  --output runs/gradcam_convnext_test
```

输出按 `original/`、`heatmap/` 和 `overlay/` 分开保存，并生成 `gradcam_manifest.csv`。

## Release 材料

Grad-CAM 审稿材料通过 GitHub Releases 发布。`reviewer-materials-v1` 标签发布于 SAM31 协议更新之前，不代表当前 main 分支；分发 SAM31 协议的 Grad-CAM 材料时应使用本次更新之后发布的 release。

## 测试

```bash
pytest -q
```
