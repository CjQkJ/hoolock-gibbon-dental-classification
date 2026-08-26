# Hoolock Dental-Image Classification: Reproducibility Code

This repository is a standalone code package for reproducing the dataset-manifest validation, offline augmentation, training and evaluation of 31 models under the reference SAM31 configuration, result aggregation, and ConvNeXt Grad-CAM analysis. The released code does not depend on the original server paths; dataset, manifest, and model-weight locations are supplied through command-line arguments.

For the Chinese documentation, see [README_ZH.md](README_ZH.md).

## Reference Configuration

The current main branch records the reference configuration `sam31_reference_v1`, using ConvNeXt-Base (`convnext_base.fb_in22k_ft_in1k`) as the primary model.

- `configs/sam31_reference.json`: reference configuration definition.
- `configs/sam31_models.lock.json`: portable integrity lock for the 31 model weights and run-time exceptions.
- `configs/models_31.json`: portable 31-model registry with relative weight paths and SHA-256 hashes.
- `results/table_s4_results.csv`: final 31-model metrics reported in Table S4.

The complete release-level protocol and reproducibility record is provided in
[`docs/SAM31_REPRODUCIBILITY.md`](docs/SAM31_REPRODUCIBILITY.md).

## Data and Experimental Configuration

- Binary classification labels: `H. hoolock` and the `H. leuconedys group`; `H. tianxing` images in the test set were assigned to the latter group.
- Curated original dataset: 340 images from 75 individuals.
- Original training data: 258 images from 60 individuals.
- Final training set: 1,984 images, comprising 258 original images, 436 museum-style augmented images, and 1,290 offline-augmented images.
- Test set: 82 original images from 15 individuals; no individual occurred in both the training and test sets.
- Online augmentation: direct square resize to each model's native input size, horizontal flipping with `p=0.5`, rotation by up to 5 degrees with `p=0.5`, and brightness jitter with an amplitude of `0.1`. The reference configuration does not use `RandomResizedCrop`.
- Optimization: AdamW with an initial learning rate of `3e-4`, weight decay of `0.01`, cosine annealing, a maximum of 200 epochs, and early-stopping patience of 30 epochs.
- SAM: non-adaptive SAM with `rho=0.05`, exact parameter restoration, and a global L2 norm over all trainable gradients.
- Loss: cross-entropy weighted inversely by training-class frequency.
- Data split: The dataset is divided into a training set and a test set for model training and evaluation.
- Input size: each model used the pretrained input resolution recorded in `configs/models_31.json`.

The logical batch size is 16. The default physical micro-batch size is 16. The audited exceptions encoded in the model registry are:

- EfficientNet-B7: input size 600, physical micro-batch 4, logical batch remains 16.
- MobileNetV5-300M: frozen backbone, physical micro-batch 4, only the classification head is trained.
- SwinV2-Large: physical micro-batch 8 on two GPUs with DataParallel.

## Important Evaluation Note

Test inference is deterministic: one view per image, one forward pass, no TTA, no calibration, no threshold tuning, no ensembling, and an `argmax` decision rule.

## Directory Structure

```text
configs/                       SAM31 protocol, model lock, and 31-model registry
metadata/dataset_manifest.csv  Portable image manifest containing 2,066 records
results/table_s4_results.csv   Final SAM31 results for the 31 models in Table S4
scripts/                       Manifest, offline augmentation, batch execution, and summary scripts
src/                           Data, transforms, models, SAM training, metrics, and Grad-CAM code
tests/                         Tests for counts, metrics, protocol metadata, and transforms
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
  --strict-expected-counts
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

The random-number state used for the historical production of the formal dataset was not retained. The release script therefore adds an explicit seed to make subsequent runs reproducible. The exact augmented files used for the reported run are fixed by the released manifest.

The 436 museum-style images were generated in advance and are managed as fixed inputs in the formal dataset; they are not generated during training. The original model used to generate these style-transferred images was not retained in the current project. Consequently, this package reproduces the complete training workflow using the released `dataset_augmented` dataset as its input.

## Training

First, check the model structure and dataset without downloading pretrained weights:

```bash
python -m src.train \
  --model-key convnext_base \
  --data-root /path/to/dataset_augmented \
  --no-pretrained --dry-run
```

For exact reproduction of the locked protocol, pass the audited server weights through `--checkpoint` or use `--checkpoint-root` in the batch script:

```bash
python -m src.train \
  --model-key convnext_base \
  --data-root /path/to/dataset_augmented \
  --checkpoint /path/to/weights/18_convnext_base.fb_in22k_ft_in1k/model.safetensors
```

If no checkpoint is supplied, timm will attempt to download pretrained weights when available. The locked weights are not stored in this repository. `models_31.json` and `sam31_models.lock.json` record the portable relative paths, file sizes, and SHA-256 hashes needed to verify a separately obtained copy of the audited weight archive `shortlist_50_20260502`.

Run the 31 models sequentially with the model-specific GPU and micro-batch settings from the registry:

```bash
nohup python scripts/run_model_zoo.py \
  --data-root /path/to/dataset_augmented \
  --checkpoint-root /path/to/weights \
  --continue-on-error > model_zoo.log 2>&1 &
```

`--checkpoint-root` is optional. When supplied, each model is started with `--checkpoint <root>/<weight_relative_path>` from `configs/models_31.json`.

## Results

`results/table_s4_results.csv` contains the final SAM31 results for all 31 models. The best model in the reference run was ConvNeXt-Base:

- Accuracy: 92.68% (76/82)
- Balanced accuracy: 90.15%
- Macro-F1: 91.55%

Rows in `results/table_s4_results.csv` are ranked by Macro-F1 in descending order, then by accuracy in descending order, and finally by Screening ID in ascending order. Accuracy is therefore not expected to be monotonic down the table.

## Grad-CAM

ConvNeXt-Base was used for morphological interpretation. The target layer was `stages.3`. By default, gradients are computed for the true class, and the complete heatmap is overlaid without applying a cutoff or mask.

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

## Release Assets

Grad-CAM comparison packages are distributed through the current GitHub
Release [`reviewer-materials-v1`](https://github.com/CjQkJ/hoolock-gibbon-dental-classification/releases/tag/reviewer-materials-v1).
The release assets use the SAM31 reference configuration and are named
`sam31_gradcam_en.zip` and `sam31_gradcam_zh.zip`. Each package contains
five model-specific galleries: the four highest-ranked SAM31 models by
Macro-F1, plus the ConvNeXt-Base no-style control. Each gallery covers the
340 original records in the source manifest (258 training-set originals and
82 test images), with portable manifests, summaries, and verification
metadata.

## Tests

```bash
pytest -q
```
