# SAM31 Reproducibility Record

## Scope and Canonical Run

This document specifies the frozen computational protocol represented by the
current `main` branch. The protocol identifier is `sam31_e73b33b_v1`; its
canonical candidate is commit `e73b33bf888e26be4833d39d776107b665e08900`
(`e73b33b`) using ConvNeXt-Base (`convnext_base.fb_in22k_ft_in1k`). The
authoritative machine-readable records are
`configs/sam31_e73b33b.json`, `configs/experiment.yaml`,
`configs/models_31.json`, and `configs/sam31_e73b33b_models.lock.json`.

For the canonical ConvNeXt-Base run, the recorded test-set result was 76/82
correct images (accuracy 92.6829%, balanced accuracy 90.1455%, and Macro-F1
91.5522%). KIZ011338 was correct for 2/2 test images and MCZ26474 was correct
for 16/16 test images. The selected checkpoint was at epoch 23.

## Dataset and Split

The portable manifest is `metadata/dataset_manifest.csv`. Its SHA-256 is
`cf9d6600c7a147fd7a2391cb179408086666255ab495977ea731eb33aad360c7`.
It contains 2,066 rows, with 1,984 training images and 82 test images.
Training images comprise 258 originals, 436 fixed museum-style images, and
1,290 offline colour-augmented images. The test set comprises 82 original
images. Training and test sets contain 60 and 15 individuals, respectively,
with no overlap in individual identifiers.

The images themselves are not included in this repository. To reproduce a
run, provide the directory containing the relative paths recorded in the
manifest through `--data-root`. Dataset access remains subject to the specimen
holding institutions and the corresponding authors.

## Model Weights and Preprocessing

The 31 registered architectures are listed in `configs/models_31.json`.
Pretrained weights are not distributed in this repository. The model registry
and integrity lock identify each audited file by the archive identifier
`shortlist_50_20260502`, a relative path, file size, and SHA-256 digest.
Supply a locally obtained copy of that archive with `--checkpoint` or
`--checkpoint-root`; the loader verifies the SHA-256 digest for registered
`.safetensors` weights and rejects non-classifier incompatibilities.

Each model uses the mean and standard deviation recorded in its own registry
entry. This is necessary because the 31 pretrained backbones do not all use
the same input normalization. Training images are directly resized to the
model-specific square input size, then undergo horizontal flipping (`p=0.5`),
rotation within plus or minus 5 degrees (`p=0.5`), and brightness jitter of
0.1. Test inference uses a direct resize, tensor conversion, and the same
model-specific normalization only.

## Training Protocol

All models use seed `20260507`, AdamW with learning rate `3e-4` and weight
decay `0.01`, `CosineAnnealingLR` with `T_max=200`, inverse-frequency weighted
cross-entropy, and non-adaptive SAM with `rho=0.05`. The maximum training
length is 200 epochs and early stopping has patience 30. cuDNN benchmarking is
disabled and deterministic mode is enabled. DataLoader workers are seeded from
the PyTorch worker seed; the documented default is four workers.

The logical batch size is 16. Physical micro-batches preserve that logical
batch during SAM gradient calculation. The documented exceptions are
EfficientNet-B7 (4), MobileNetV5-300M (4 with frozen backbone), and
SwinV2-Large (8 using two GPUs with DataParallel). All other models start with
a physical micro-batch of 16 on one GPU.

## Evaluation Contract and Interpretation

Test inference uses one view and one forward pass per image, an `argmax`
decision rule, and no test-time augmentation, calibration, threshold tuning,
ensembling, or metadata-based post-processing. Results in
`results/table_s4_results.csv` are sorted by descending Macro-F1, then
descending accuracy, then ascending Screening ID.

No separate validation split was used in the historical SAM31 experiment.
The 82-image test set was monitored after each epoch for early stopping and
checkpoint selection. Consequently, these values describe a test-guided
model-selection protocol and should not be interpreted as performance on a
fully independent holdout set. The code preserves this historical procedure
for auditability and exposes `selection_split: test` in effective run records.

## Reproduction Commands

Run a structural check without downloading a pretrained model:

```bash
python -m src.train \
  --model-key convnext_base \
  --data-root /path/to/dataset_augmented \
  --no-pretrained --dry-run
```

Run the canonical architecture with a verified audited pretrained weight:

```bash
python -m src.train \
  --model-key convnext_base \
  --data-root /path/to/dataset_augmented \
  --checkpoint /path/to/weights/18_convnext_base.fb_in22k_ft_in1k/model.safetensors
```

Run the registered 31-model set sequentially:

```bash
python scripts/run_model_zoo.py \
  --data-root /path/to/dataset_augmented \
  --checkpoint-root /path/to/weights \
  --continue-on-error
```

The historical GitHub release `reviewer-materials-v1` predates this frozen
SAM31 protocol and is not a reproduction artifact for the current branch.
