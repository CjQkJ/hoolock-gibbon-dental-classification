# 白眉长臂猿牙齿图像分类：论文复现代码

本目录是面向论文审稿的独立代码包，覆盖数据清单校验、离线增强、31 个模型训练与评估、结果汇总，以及 ConvNeXt Grad-CAM。代码不依赖原服务器路径，数据和权重均通过命令行参数传入。

英文说明见 [README.md](README.md)。

## 论文实验口径

- 二分类标签：`H. hoolock` 与 `H. leuconedys group`；测试集中的 `H. tianxing` 并入后者。
- 清理后原始数据：75 个体、340 张图片。
- 训练原图：258 张、60 个体。
- 最终训练集：1984 张，包括 258 张原图、436 张博物馆风格化图像和 1290 张离线增强图像。
- 测试集：82 张原图、15 个体；个体不跨训练集和测试集。
- 在线增强：`RandomResizedCrop(0.85-1.00, ratio=1:1)`、水平翻转 `p=0.5`、5 度旋转 `p=0.5`、亮度抖动 `0.1`。
- 优化：AdamW，初始学习率 `3e-4`，权重衰减 `0.01`，余弦退火，最多 200 epoch，早停 patience 30。
- 损失：按训练类别频数倒数加权的交叉熵。
- 选择指标：Macro-F1。
- 输入大小：各模型使用 `configs/models_31.json` 记录的预训练分辨率。

## 重要评估说明

原实验没有单独验证集。测试集同时用于逐 epoch 的早停、最佳 checkpoint 选择和最终指标报告。因此结果是“测试集参与模型选择”的实验结果，可能比完全独立测试评估乐观。发布代码忠实保留这一流程，并在 `selection_split: test` 中显式记录。

## 文件结构

```text
configs/                    统一实验参数和 31 模型清单
metadata/dataset_manifest.csv  可移植的 2066 行图像清单
results/table_s4_results.csv   论文 Table S4 的 31 模型结果
scripts/                    清单、离线增强、批量运行和汇总脚本
src/                        数据、变换、模型、训练、指标和 Grad-CAM
tests/                      计数、指标、配置和变换测试
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

436 张博物馆风格化图像是预先生成并随正式数据集管理的固定输入，不在训练时生成。当前工程中没有保存其原始风格化生成模型，因此本包复现的是以发布版 `dataset_augmented` 为输入的完整训练流程。

## 训练

先进行不下载权重的结构和数据检查：

```bash
python -m src.train \
  --model-key convnext_base \
  --data-root /path/to/dataset_augmented \
  --no-pretrained --dry-run
```

训练 ConvNeXt-Base：

```bash
python -m src.train \
  --model-key convnext_base \
  --data-root /path/to/dataset_augmented
```

依次运行 31 个模型：

```bash
nohup python scripts/run_model_zoo.py \
  --data-root /path/to/dataset_augmented \
  --continue-on-error > model_zoo.log 2>&1 &
```

大多数模型使用 batch size 16。实际结果中的两个例外已写入模型配置：EfficientNet-B7 使用 batch size 4；MobileNetV5-300M 使用 micro-batch 4、4 步梯度累积、学习率 `1e-5`，冻结 backbone，仅训练分类头。

31 模型中 EfficientNet-B3 的数值性能最高；ConvNeXt-Base 因保持较高分类性能且 Grad-CAM 在牙冠区域的定位更清晰，被选作论文形态解释模型。

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

## 测试

```bash
pytest -q
```
