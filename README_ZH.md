# 白眉长臂猿牙齿图像分类：可复现代码

本仓库是用于复现数据清单校验、离线增强、按照 SAM31 参考配置训练和评估 31 个模型、结果汇总以及 ConvNeXt Grad-CAM 的独立代码包。代码不依赖原服务器路径，数据集、清单和模型权重位置均通过命令行参数传入。

英文说明见 [README.md](README.md)。

## 参考配置

当前 main 分支记录参考配置 `sam31_reference_v1`，并以 ConvNeXt-Base（`convnext_base.fb_in22k_ft_in1k`）作为主要模型。

- `configs/sam31_reference.json`：参考配置定义。
- `configs/sam31_models.lock.json`：31 个模型权重完整性和运行例外的可移植锁定文件。
- `configs/models_31.json`：可移植的 31 模型注册表，含相对权重路径和 SHA-256 哈希。
- `results/table_s4_results.csv`：论文 Table S4 使用的最终 31 模型指标。

完整的发布级协议和复现记录见
[`docs/SAM31_REPRODUCIBILITY.md`](docs/SAM31_REPRODUCIBILITY.md)。

## 数据与实验配置

- 二分类标签：`H. hoolock` 与 `H. leuconedys group`；测试集中的 `H. tianxing` 并入后者。
- 清理后原始数据：75 个体、340 张图片。
- 训练原图：258 张、60 个体。
- 最终训练集：1984 张，包括 258 张原图、436 张博物馆风格化图像和 1290 张离线增强图像。
- 测试集：82 张原图、15 个体；个体不跨训练集和测试集。
- 在线增强：直接正方形缩放至各模型原生输入尺寸、水平翻转 `p=0.5`、5 度旋转 `p=0.5`、亮度抖动 `0.1`。参考配置不使用 `RandomResizedCrop`。
- 优化：AdamW，初始学习率 `3e-4`，权重衰减 `0.01`，余弦退火，最多 200 epoch，早停 patience 30。
- SAM：非自适应 SAM，`rho=0.05`，参数精确恢复，对所有可训练梯度计算全局 L2 范数。
- 损失：按训练类别频数倒数加权的交叉熵。
- 数据划分：数据集分为训练集和测试集，用于模型训练和评估。
- 输入大小：各模型使用 `configs/models_31.json` 记录的预训练分辨率。

逻辑 batch size 为 16，默认物理 micro-batch 为 16。模型注册表记录的审计例外如下：

- EfficientNet-B7：输入 600，物理 micro-batch 4，逻辑 batch 仍为 16。
- MobileNetV5-300M：冻结 backbone，物理 micro-batch 4，仅训练分类头。
- SwinV2-Large：物理 micro-batch 8、两卡 DataParallel。

## 重要评估说明

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
  --strict-expected-counts
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

未传 checkpoint 时，timm 会在可用时尝试下载预训练权重。锁定权重不存储在本仓库中；`models_31.json` 和 `sam31_models.lock.json` 记录相对路径、文件大小和 SHA-256，可用于核验另行取得的审计权重归档 `shortlist_50_20260502`。

使用注册表里的模型专用 GPU 和 micro-batch 设置依次运行 31 个模型：

```bash
nohup python scripts/run_model_zoo.py \
  --data-root /path/to/dataset_augmented \
  --checkpoint-root /path/to/weights \
  --continue-on-error > model_zoo.log 2>&1 &
```

`--checkpoint-root` 可选；传入后，每个模型会以 `--checkpoint <root>/<weight_relative_path>` 启动，路径来自 `configs/models_31.json`。

## 结果

`results/table_s4_results.csv` 包含全部 31 个模型的最终 SAM31 结果。参考运行中的最优模型为 ConvNeXt-Base：

- Accuracy：92.68%（76/82）
- Balanced accuracy：90.15%
- Macro-F1：91.55%

`results/table_s4_results.csv` 按 Macro-F1 降序排列；Macro-F1 相同则按 Accuracy 降序排列，仍相同则按 Screening ID 升序排列。因此，表中 Accuracy 不一定从上至下单调下降。

## Grad-CAM

研究使用 ConvNeXt-Base 进行形态解释。目标层为 `stages.3`，默认对真实类别求梯度，使用完整热力图覆盖，不设置 cutoff 或 mask。

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

Grad-CAM 对比材料通过当前的 GitHub
Release [`reviewer-materials-v1`](https://github.com/CjQkJ/hoolock-gibbon-dental-classification/releases/tag/reviewer-materials-v1)
发布。该 Release 的附件采用 SAM31 参考配置，文件名为
`sam31_gradcam_en.zip` 和 `sam31_gradcam_zh.zip`。每个压缩包包含
5 个模型画廊：SAM31 中按 Macro-F1 排列的前 4 个模型，以及 ConvNeXt-Base
无风格化对照。每个画廊覆盖源 manifest 中的 340 条原始记录（训练集原图 258
张、测试图 82 张），并包含可移植的 manifest、汇总和核验元数据。

## 测试

```bash
pytest -q
```
