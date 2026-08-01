# Grad-CAM Reviewer Materials

This release contains Grad-CAM comparison materials for the SAM31 reference
configuration.

Assets:

- `sam31_gradcam_en.zip`
- `sam31_gradcam_zh.zip`

Each package contains galleries for the four highest-ranked SAM31 models by
Macro-F1 and the ConvNeXt-Base no-style control. Each gallery covers the 340
original records in the source manifest (258 training-set originals and 82
test images), producing 1,700 Grad-CAM images in total. The packages
include portable file manifests, model-level results, configuration metadata,
and SHA-256 checksums. Absolute server and local Windows paths have been
removed.

Grad-CAM was generated with the true class as the target and the complete
heatmap overlaid on the image. No activation cutoff, mask, test-time
augmentation, threshold tuning, calibration, or ensembling was used.

The reference ConvNeXt-Base run recorded 92.68% accuracy, 90.15% balanced
accuracy, and 91.55% Macro-F1.

## 中文说明

本 Release 提供 SAM31 Grad-CAM 对比材料的中英文版本：

- `sam31_gradcam_en.zip`：英文版
- `sam31_gradcam_zh.zip`：中文版

每个压缩包均包含按 Macro-F1 排列的前 4 个 SAM31 模型和 ConvNeXt-Base
无风格化对照。每个模型页面覆盖同一份 340 张原始牙齿图像清单，其中包括 258
张训练集原图和 82 张测试图，共包含 1,700 张 Grad-CAM 图。包内路径均为相对
路径，并附有清单、模型级结果、配置元数据和 SHA-256 校验和。

Grad-CAM 以真实类别为目标，使用完整热力图叠加；未使用激活阈值、掩膜、测试时
增强、阈值调节、校准或模型集成。
