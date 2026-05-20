# Reflection Report — CIV1287H Vibe-Coding Assignment

*Author: <your name> &nbsp;|&nbsp; Student #: <your #> &nbsp;|&nbsp; Date: 2026-05-19*

## Q1. AI-Assisted Development Experience (Vibe Coding)
*How did using AI-assisted tools change the way you built your solution?*

Using AI-assisted tools shifted my workflow from "find a template and adapt it" to "describe what I want and review what's produced." Normally for a project like this I would search GitHub or Stack Overflow for a similar transfer-learning example and reshape it around my data. Here I gave the assistant my hardware context (RTX 3050 Ti laptop, 4 GB VRAM, Linux Mint), the Kaggle dataset link, and the assignment constraints, and it produced a working pipeline I could iterate on. I spent far more time reading and reviewing generated code than writing it, and the loop became: specify intent → read the output → run it → correct or refine where it broke.

## Q2. Where AI-Assisted Building Worked Well
*In which parts of your workflow did AI-assisted tools help the most?*

The biggest wins were in the engineering "plumbing" around the model. PyTorch transfer-learning setup — loading a pre-trained ResNet18 and MobileNetV3-Small, freezing the backbone, swapping the classifier head, running mixed-precision training with AMP/GradScaler — came out correctly with very little back-and-forth, which would have taken me hours of doc-reading otherwise. The same was true for the evaluation outputs: confusion matrices, per-class precision/recall/F1 tables, and the Word-table export of all metrics. Environment setup was the other clear win — the assistant unblocked the venv without sudo, switched from the Kaggle CLI to `kagglehub` when authentication was awkward, and untangled the nested folder structure inside the dataset zip.

## Q3. Where AI-Assisted Building Failed or Was Limited
*Where did the AI-assisted approach not work well or require correction?*

The two biggest gaps were hardware-aware decisions and the upfront "what dataset do I even use" question. The assistant couldn't see my GPU, so calls about backbone choice (ResNet18 vs. anything deeper), batch size, image resolution, and how much of the network to fine-tune all came back to me weighing what 4 GB of VRAM would tolerate without OOM'ing. Finding a suitable dataset was also on me: I had to pick a road-defect dataset that matched the construction-engineering framing of the course, locate it on Kaggle, and confirm it had enough images per class for transfer learning to be meaningful. Once I supplied those decisions and the dataset link, the AI ran with them — but it would not have arrived there on its own.

## Q4. Understanding vs Dependence
*Which parts of the system did you fully understand, and which parts did you rely on AI tools to complete?*

Honestly, mostly borrowed. I understand the higher-level concepts — what transfer learning is doing, why a confusion matrix is a more useful diagnostic than a single accuracy number, why we normalize with ImageNet mean/std, why a fixed-seed 70/15/15 split matters for reproducibility, and why class-balanced loss weighting can matter on a small dataset. But I would struggle to write the actual training loop, the AMP/`autocast` scaffolding, or the sklearn/seaborn evaluation plotting from scratch in a blank editor. If I had to reproduce this pipeline without an assistant, I could re-run and tweak the existing code, but I could not rebuild it from memory — and I think being honest about that is the point of this question.

## Q5. Engineering Judgment and Future Use
*Why is human judgment still important even when AI tools can produce working solutions? How would you realistically use these tools in future engineering work?*

In real engineering work the model returns labels, not decisions, and I would use these tools accordingly. AI-assisted development is excellent for getting a first working prototype quickly — useful for screening whether an approach is even viable on a given dataset — but the output then has to be validated against data that matches the real deployment domain (Toronto pavement, the lighting and camera angles of an actual inspection vehicle, seasonal variation) before anyone should trust it for maintenance prioritization. I would also use these tools as a pair-programmer for unfamiliar libraries: PyTorch, GIS toolkits, anything with a large API surface where asking-and-verifying is faster than reading docs end-to-end. The judgment calls stay with the engineer: which metric matters most, what false-negative rate is acceptable when a missed pothole carries a real safety and liability cost, and whether 83 % accuracy is or is not "good enough" for the workflow it is being deployed into.
