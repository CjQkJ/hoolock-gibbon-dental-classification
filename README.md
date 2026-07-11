# Hoolock Dental-Image Classification: Reproducibility Code for the Manuscript

This directory contains a standalone code package prepared for manuscript review. It covers dataset-manifest validation, offline augmentation, training and evaluation of 31 models, result aggregation, and ConvNeXt Grad-CAM analysis. The code does not depend on paths from the original server; dataset and weight locations are supplied through command-line arguments.

For the Chinese documentation, see [README_ZH.md](README_ZH.md).

## Experimental Scope of the Manuscript

- Binary classification labels: `H. hoolock` and the `H. leuconedys group`; `H. tianxing` images in the test set were assigned to the latter group.
- Curated original dataset: 340 images from 75 individuals.
- Original training data: 258 images from 60 individuals.
- Final training set: 1,984 images, comprising 258 original images, 436 museum-style augmented images, and 1,290 offline-augmented images.
- Test set: 82 original images from 15 individuals; no individual occurred in both the training and test sets.
- Online augmentation: `RandomResizedCrop(0.85-1.00, ratio=1:1)`, horizontal flipping with `p=0.5`, rotation by up to 5 degrees with `p=0.5`, and brightness jitter with an amplitude of `0.1`.
- Optimization: AdamW with an initial learning rate of `3e-4`, weight decay of `0.01`, cosine annealing, a maximum of 200 epochs, and an early-stopping patience of 30 epochs.
- Loss: cross-entropy weighted inversely by training-class frequency.
- Model-selection metric: Macro-F1.
- Input size: each model used the pretrained input resolution recorded in `configs/models_31.json`.

## Important Evaluation Note

The original experiment did not use a separate validation set. The test set was evaluated after every epoch and was used for early stopping, best-checkpoint selection, and final metric reporting. The reported results therefore reflect a protocol in which the test set participated in model selection and may be optimistic relative to evaluation on a fully independent test set. The release code faithfully preserves this procedure and records it explicitly as `selection_split: test`.

## Directory Structure

```text
configs/                       Unified experiment parameters and the 31-model registry
metadata/dataset_manifest.csv  Portable image manifest containing 2,066 records
results/table_s4_results.csv   Results for the 31 models reported in Table S4
scripts/                       Manifest, offline augmentation, batch execution, and summary scripts
src/                           Data, transforms, models, training, metrics, and Grad-CAM code
tests/                         Tests for counts, metrics, configuration, and transforms
```

## Environment

The original experiments used PyTorch `2.10.0+cu128`, CUDA `12.8`, cuDNN `9.10.2`, and timm `1.0.25` on NVIDIA GeForce RTX 4090 GPUs with 24 GB of memory. According to the run records, 30 models were trained on a single GPU, whereas SwinV2-Large used two GPUs with DataParallel. Example installation commands are shown below:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0
pip install -r requirements.txt
```

## Dataset Directory

The code expects the following directory structure. Additional intermediate directories representing style sources and individual specimens are permitted:

```text
dataset_augmented/
  train/
    hoolock/**/*.jpg
    leuconedys/**/*.jpg
  test/
    hoolock/**/*.jpg
    leuconedys/**/*.jpg
```

Validate the released dataset and rebuild the manifest as follows:

```bash
python scripts/build_manifest.py \
  --data-root /path/to/dataset_augmented \
  --output metadata/dataset_manifest.csv \
  --strict-paper-counts
```

Strict validation should report `966/1018` training images and `28/54` test images for the two class labels, with `60/15` individuals in the training and test sets, respectively.

## Offline Augmentation

For each original training image, five transformations were sampled without replacement from eight colour-profile presets: `bright_sat`, `dark_con`, `warm`, `cool`, `hicon`, `soft`, `vivid`, and `muted`. Complete parameter ranges are recorded in `scripts/offline_augmentation.py`.

```bash
python scripts/offline_augmentation.py \
  --source-root /path/to/original_train_test \
  --output-root /path/to/generated_dataset \
  --seed 20260430
```

The random-number state used for the historical production of the formal dataset was not retained. The release script therefore adds an explicit seed to make subsequent runs reproducible. The exact augmented files used for the manuscript are fixed by the released manifest.

The 436 museum-style images were generated in advance and are managed as fixed inputs in the formal dataset; they are not generated during training. The original model used to generate these style-transferred images was not retained in the current project. Consequently, this package reproduces the complete training workflow using the released `dataset_augmented` dataset as its input.

## Training

First, check the model structure and dataset without downloading pretrained weights:

```bash
python -m src.train \
  --model-key convnext_base \
  --data-root /path/to/dataset_augmented \
  --no-pretrained --dry-run
```

Train ConvNeXt-Base:

```bash
python -m src.train \
  --model-key convnext_base \
  --data-root /path/to/dataset_augmented
```

Run the 31 models sequentially:

```bash
nohup python scripts/run_model_zoo.py \
  --data-root /path/to/dataset_augmented \
  --continue-on-error > model_zoo.log 2>&1 &
```

Most models used a batch size of 16. Two exceptions in the actual run records are encoded in the model configuration: EfficientNet-B7 used a batch size of 4; MobileNetV5-300M used a micro-batch size of 4, four gradient-accumulation steps, a learning rate of `1e-5`, and a frozen backbone, with only the classification head trained.

Among the 31 models, EfficientNet-B3 achieved the best numerical performance. ConvNeXt-Base was selected for morphological interpretation because it retained high classification performance and produced cleaner Grad-CAM localization over the tooth crown.

## Grad-CAM

ConvNeXt-Base was used for morphological interpretation in the manuscript. The target layer was `stages.3`. By default, gradients are computed for the true class, and the complete heatmap is overlaid without applying a cutoff or mask.

```bash
python -m src.gradcam \
  --checkpoint /path/to/convnext_best_model.pth \
  --data-root /path/to/dataset_augmented \
  --split test \
  --source-type original \
  --target true \
  --output runs/gradcam_convnext_test
```

Outputs are stored separately under `original/`, `heatmap/`, and `overlay/`, together with a `gradcam_manifest.csv` file.

## Supplementary Grad-CAM Packages

The complete reviewer packages are available from the [Grad-CAM Reviewer Materials release](https://github.com/CjQkJ/hoolock-gibbon-dental-classification/releases/tag/reviewer-materials-v1):

- [English Grad-CAM package](https://github.com/CjQkJ/hoolock-gibbon-dental-classification/releases/download/reviewer-materials-v1/GradCAM_English.zip)
- [Chinese Grad-CAM package with results](https://github.com/CjQkJ/hoolock-gibbon-dental-classification/releases/download/reviewer-materials-v1/GradCAM_Chinese_with_results.zip)

## Tests

```bash
pytest -q
```
