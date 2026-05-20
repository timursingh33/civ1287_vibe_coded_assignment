# CLAUDE.md — Road Defect Classification Project

## Project Overview

This project is a coursework submission for **CIV1287H: Virtualization and Analytics in Construction** (University of Toronto). The goal is to build an AI-assisted workflow for **road defect classification** using pre-trained deep learning models, following the "vibe coding" style required by the assignment: assemble pre-trained components into a working pipeline rather than build models from scratch.

The construction-engineering relevance is **pavement condition assessment** — automatically classifying road surface defects (cracks, potholes, etc.) is directly applicable to infrastructure inspection, maintenance planning, and asset management.

## Assignment Requirements (30 marks total)

### Part A — Implementation (10 marks)
- Notebook with a clear **input → processing → output** flow
- Use a **pre-trained AI tool** (no training a model from scratch)
- Test variations (different inputs / settings / scenarios)
- Document inputs and outputs clearly

### Part B — Evaluation (10 marks)
- **Qualitative**: Do outputs make sense? Are they consistent across inputs? Useful for the construction application?
- **Quantitative**: Number of test cases, correct vs incorrect counts, accuracy %, true positives, false positives, false negatives, per-class metrics

### Part C — Reflection (10 marks, ~1 page)
Five reflection questions on the vibe-coding experience, where AI-assisted building helped vs failed, understanding vs dependence, and the role of engineering judgment.

### Deliverables
1. Implementation notebook (`.ipynb`)
2. Evaluation report — 1 page
3. Reflection report — 1 page, answering all five Part C questions

## Dataset

- **Source**: https://www.kaggle.com/datasets/patelmihir/road-defects-nonaugmented
- **Task**: Image classification of road defects
- **Download** via Kaggle CLI:
  ```bash
  pip install kaggle
  # Put your kaggle.json API token in ~/.kaggle/ and chmod 600 it
  kaggle datasets download -d patelmihir/road-defects-nonaugmented
  unzip road-defects-nonaugmented.zip -d data/
  ```

After downloading, **first task** is to inspect the folder structure and class distribution before writing any model code — class imbalance matters for the evaluation metrics.

## Hardware & Environment

- **OS**: Linux Mint
- **GPU**: NVIDIA RTX 3050 laptop (≈4 GB VRAM; possibly Ti variant — confirm with `nvidia-smi`)
- **Implication**: Small VRAM budget. Prefer:
  - Smaller backbones: MobileNetV3, EfficientNet-B0, ResNet18, or ResNet50 at most
  - Image size 224×224
  - Batch size 16–32 (drop to 8 if you hit OOM)
  - Mixed-precision training (`torch.cuda.amp`) to roughly halve memory use
  - `num_workers=2–4` for the DataLoader (laptop CPU)

### Verify GPU before anything else
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Recommended environment setup
```bash
python3 -m venv .venv
source .venv/bin/activate
# CUDA 12.1 wheels work on most modern Linux Mint installs; adjust if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install jupyterlab matplotlib scikit-learn seaborn pandas pillow tqdm
```

## Recommended Workflow

Use **transfer learning** with an ImageNet pre-trained model — this is the cleanest match for the "use pre-trained AI tools" requirement.

1. **Input section**
   - Load the dataset, list class names, count images per class
   - Display 1–2 sample images per class (sanity check)
   - Split into train / val / test (e.g., 70 / 15 / 15) with a fixed random seed

2. **Processing section**
   - Load pre-trained model: `torchvision.models.resnet18(weights="IMAGENET1K_V1")`
   - Replace the final layer with `nn.Linear(in_features, num_classes)`
   - Freeze backbone, train classifier head for a few epochs
   - Optionally unfreeze the last block for a short fine-tuning pass at a lower LR
   - Use `CrossEntropyLoss` (with `weight=` for class imbalance if needed) and the Adam optimizer

3. **Output section**
   - Predictions on the held-out test set
   - **Confusion matrix** + per-class precision / recall / F1 (use `sklearn.metrics`)
   - Display a grid of correct and incorrect predictions (qualitative evidence)
   - Save metrics to `outputs/metrics.json` so the evaluation report can pull from real numbers

## Variations to Test (Part A requirement)

Pick at least one of:
- Two backbones compared (e.g., ResNet18 vs MobileNetV3-Small)
- Two image resolutions (224 vs 128)
- With vs without data augmentation (flips, rotations, color jitter)
- Frozen backbone vs partially-unfrozen fine-tuning

Tabulate results so the "comparison across input conditions" objective is visibly satisfied.

## Suggested Project Structure

```
.
├── CLAUDE.md
├── README.md
├── .gitignore                      # ignore data/, .venv/, outputs/checkpoints/
├── data/                           # Kaggle dataset (do not commit)
├── notebooks/
│   └── road_defect_classification.ipynb
├── outputs/
│   ├── figures/                    # confusion matrices, sample predictions
│   ├── metrics.json                # all quantitative numbers used in report
│   └── checkpoints/                # model weights
└── reports/
    ├── evaluation_report.md        # 1 page
    └── reflection_report.md        # 1 page
```

## Key Constraints (do not violate)

- **Use pre-trained models only** — no hand-built CNN architectures
- **Keep the notebook readable** — graded on engineering thinking, not algorithmic complexity
- **Every cell should have a short markdown explanation** above it
- **Record numbers as they're produced** — don't try to reconstruct evaluation metrics from memory later
- **Set random seeds** so results are reproducible
- **Training time on the RTX 3050 should stay reasonable** (target <30 min per experiment); if it doesn't, shrink the model or freeze more layers

## What Claude Code Should Help With

- Writing notebook cells in the input → processing → output pattern
- Setting up PyTorch transfer learning correctly for a 4 GB GPU
- Generating evaluation metrics, confusion matrices, and figures
- Drafting the evaluation report from the actual numbers in `outputs/metrics.json`
- Debugging CUDA / VRAM issues, dataloader bottlenecks, version mismatches

## What I (the student) Must Do Myself

- The **reflection report** — must reflect my personal experience with the five Part C questions
- Final interpretation of the results in the evaluation report
- The engineering-judgment commentary: which defect classes matter most for road maintenance, what the false-negative rate implies for real inspection workflows, etc.
- Deciding what counts as "good enough" for this construction application
