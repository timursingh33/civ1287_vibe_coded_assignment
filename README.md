# CIV1287H — Road Defect Classification (Vibe-Coding Assignment)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timursingh33/civ1287_vibe_coded_assignment/blob/main/notebooks/road_defect_classification.ipynb)

Transfer-learning pipeline for road-defect image classification using pre-trained ImageNet backbones.
Built locally on Linux Mint + NVIDIA RTX 3050 Ti (4 GB VRAM); reproducible in Google Colab.

## Contents

- `notebooks/road_defect_classification.ipynb` — main implementation (input → processing → output)
- `notebooks/highres_448/road_defect_classification_448.ipynb` — 448×448 high-resolution variation
- `outputs/metrics.json` + `outputs/figures/` — quantitative results and figures
- `reports/evaluation_report.md` — Part B evaluation report
- `reports/reflection_report.md` — Part C reflection report

## Dataset

[Kaggle: patelmihir/road-defects-nonaugmented](https://www.kaggle.com/datasets/patelmihir/road-defects-nonaugmented)
— 4 classes (Cracks, Patch, Potholes, Surface_Defects), 100 images each, ~391 MB total.

The dataset is not redistributed in this repo. Download it directly from Kaggle:

```python
import kagglehub
DATA_DIR = kagglehub.dataset_download("patelmihir/road-defects-nonaugmented")
print(DATA_DIR)
```

This requires a Kaggle API token (`~/.kaggle/kaggle.json`). Get one at
https://www.kaggle.com/settings → "Create New API Token". The Kaggle archive ships each
class nested in a same-named subfolder; the notebook handles that nesting before passing
the path to `ImageFolder`.

## Reproduce locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install jupyterlab matplotlib scikit-learn seaborn pandas pillow tqdm kagglehub ipykernel
jupyter lab notebooks/road_defect_classification.ipynb
```

Run the cells top-to-bottom. The notebook saves metrics to `outputs/metrics.json` and figures
to `outputs/figures/`.

## Reproduce in Google Colab

Click the **Open in Colab** badge above. In the first runnable cell, provide your Kaggle
credentials (either upload `kaggle.json` via `google.colab.files.upload()`, or set them
through Colab Secrets as `KAGGLE_USERNAME` / `KAGGLE_KEY`). Then `Runtime → Change runtime
type → T4 GPU` and run all.

## Hardware notes

- Image size 224×224, batch 32, mixed-precision (AMP). Training fits comfortably in 4 GB VRAM.
- Two backbones compared: **ResNet18** (primary) and **MobileNetV3-Small** (variation required by Part A).
- All randomness is seeded (`SEED=42`); re-running reproduces the same splits and metrics.

## Final results

| Model | Test accuracy | Macro F1 | Train time (s) |
|---|---|---|---|
| ResNet18 (head + ft) | 83.3% | 0.833 | 47 |
| MobileNetV3-Small (head + ft) | 78.3% | 0.784 | 49 |

Held-out test set: 60 images. Full metrics and confusion matrices in `outputs/`.
