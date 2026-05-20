# CIV1287H — Road Defect Classification (Vibe-Coding Assignment)

Transfer-learning pipeline for road-defect image classification using pre-trained ImageNet backbones.
Built locally on Linux Mint + NVIDIA RTX 3050 Ti (4 GB VRAM).

**Deliverables**
- `notebooks/road_defect_classification.ipynb` — implementation notebook (input → processing → output)
- `outputs/metrics.json` + `outputs/figures/` — quantitative results and figures
- `reports/evaluation_report.md` — auto-generated from `outputs/metrics.json` (Part B, 1 page)
- `reports/reflection_report.md` — student-authored reflection (Part C, 1 page)

## Reproduce

```bash
# 1. environment
python3 -m venv .venv --without-pip
.venv/bin/python <(curl -sS https://bootstrap.pypa.io/get-pip.py)
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple torch torchvision
.venv/bin/pip install jupyterlab matplotlib scikit-learn seaborn pandas \
    pillow tqdm kagglehub ipykernel

# 2. data (≈383 MB)
.venv/bin/python scripts/download_data.py

# 3. notebook
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    notebooks/road_defect_classification.ipynb

# 4. evaluation report (re-runnable any time after the notebook produces metrics.json)
.venv/bin/python scripts/build_evaluation_report.py
```

## Dataset

[Kaggle: patelmihir/road-defects-nonaugmented](https://www.kaggle.com/datasets/patelmihir/road-defects-nonaugmented)
— 4 classes (Cracks, Patch, Potholes, Surface_Defects), 100 images each.

The Kaggle archive ships each class nested in a same-named subfolder; `scripts/download_data.py`
flattens that layout into `data/road_defects/<class>/<images>` via symlinks for `ImageFolder`.

## Hardware notes

- Image size 224×224, batch 32, mixed-precision (AMP). Training fits comfortably in 4 GB VRAM.
- Two backbones compared: **ResNet18** (primary) and **MobileNetV3-Small** (variation required by Part A).
- All randomness is seeded (`SEED=42`); re-running reproduces the same splits and metrics.
