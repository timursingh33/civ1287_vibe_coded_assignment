# Evaluation Report — Road Defect Classification

*Author: <your name> &nbsp;|&nbsp; CIV1287H &nbsp;|&nbsp; Generated from `outputs/metrics.json`*

## System summary
We fine-tuned an **ImageNet-pretrained ResNet18** for 4-class road-defect image classification on the Kaggle *patelmihir/road-defects-nonaugmented* dataset. The backbone was frozen for 5 head-only epochs, then the last block was unfrozen for 3 fine-tuning epochs at a lower learning rate. As the required variation we re-ran the same recipe with **MobileNetV3-Small** to compare a smaller, edge-deployable backbone.

- Dataset: 400 images, 4 classes (`Cracks, Patch, Potholes, Surface_Defects`)
- Split: train **280** / val **60** / test **60**
- Image size 224×224, batch 32, seed 42

## Quantitative results

| Model | Test accuracy | Macro F1 | Weighted F1 | Train time (s) |
|---|---|---|---|---|
| ResNet18 (head + ft) | 83.3% | 0.833 | 0.831 | 47 |
| MobileNetV3-Small (head + ft) | 78.3% | 0.784 | 0.787 | 49 |

- Test cases: **60**
- Correct (primary): **50**
- Incorrect (primary): **10**

### Per-class metrics — ResNet18 (head + ft)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cracks | 0.786 | 0.786 | 0.786 | 14 |
| Patch | 0.875 | 0.933 | 0.903 | 15 |
| Potholes | 0.800 | 0.923 | 0.857 | 13 |
| Surface_Defects | 0.867 | 0.722 | 0.788 | 18 |

### Confusion matrix — ResNet18 (head + ft)
| true ↓ \ pred → | Cracks | Patch | Potholes | Surface_Defects |
|---|---|---|---|---|
| **Cracks** | 11 | 0 | 2 | 1 |
| **Patch** | 0 | 14 | 0 | 1 |
| **Potholes** | 1 | 0 | 12 | 0 |
| **Surface_Defects** | 2 | 2 | 1 | 13 |

Largest confusion cell: **2** images of class *Cracks* were predicted as *Potholes*.

## Qualitative observations
- Predicted labels were consistent with the visual content for the majority of test images (see `outputs/figures/qualitative_correct.png`).
- Most errors involved visually overlapping classes — typically cracks vs. surface defects, where the texture cues are subtle (see `outputs/figures/qualitative_wrong.png`).
- Outputs are reproducible from seed 42; re-running the notebook produces the same split and the same per-batch order.

## Strengths
- Transfer learning from ImageNet works well on this small dataset; head-only training reaches usable accuracy within a few epochs on a 4 GB GPU.
- The pipeline is **deterministic** (fixed seed) and **fits the hardware budget** (mixed-precision, 224×224, batch 32, training under 49 s).
- All quantitative numbers in this report are generated from `outputs/metrics.json`, so the report cannot drift from the actual experiment outputs.

## Limitations
- Only **400 images total** — accuracy on the held-out test set (60 images) is high-variance. A larger test set would be needed to make confident claims about deployment performance.
- The dataset is curated, well-lit, and cropped; real Toronto pavement images have motion blur, shadows, wet surfaces, and partial occlusion, which the model has not seen.
- The classes are visually overlapping (cracks can co-occur with surface defects and patches), so a single-label classifier is a simplification — a multi-label or detection model would be a better long-term fit for an inspection workflow.

## Engineering interpretation
For pavement-condition assessment, **false negatives** (missing a real defect) are typically worse than false positives, because they delay maintenance and let damage compound. The per-class **recall** column is therefore the most important number to scrutinise: see how *Potholes* and *Cracks* recall compare against *Patch* and *Surface_Defects*. The MobileNetV3 comparison shows whether the smaller backbone could plausibly run on a vehicle-mounted edge device with the same accuracy ceiling.
