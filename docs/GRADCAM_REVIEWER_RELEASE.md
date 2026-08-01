# Grad-CAM Reviewer Materials

This release contains the audited Grad-CAM reviewer materials for the frozen
`SAM31/e73b33b` protocol.

Assets:

- `GradCAM_SAM31_e73b33b_English.zip`
- `GradCAM_SAM31_e73b33b_Chinese.zip`

Each package contains 33 original dental images and 132 Grad-CAM overlays for
four selected models and four selected individuals. Original images and
overlays are separated and grouped by tooth position. The packages include
portable file manifests, model-level results, protocol metadata, and SHA-256
checksums. Absolute server and local Windows paths have been removed.

Grad-CAM was generated with the true class as the target and the complete
heatmap overlaid on the image. No activation cutoff, mask, test-time
augmentation, threshold tuning, calibration, or ensembling was used.

The canonical ConvNeXt-Base run recorded 92.68% accuracy, 90.15% balanced
accuracy, and 91.55% Macro-F1. KIZ011338 was correct for 2/2 test images and
MCZ26474 was correct for 16/16 test images.
