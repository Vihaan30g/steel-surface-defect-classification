# Steel Surface Defect Classification using Transfer Learning

Deep learning based steel surface defect classification system.


---

## Project Overview

This project classifies different types of steel surface defects using convolutional neural networks and transfer learning.

The final model uses a pretrained ResNet18 architecture fine-tuned on the NEU Surface Defect Database and achieves near-perfect validation performance.

---

## Defect Classes

The model classifies the following six defect categories:

- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

---

## Dataset

Dataset used:

NEU Surface Defect Database

Kaggle Dataset Link:

https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database/data

---

## Final Model

Architecture:
- ResNet18 Transfer Learning

Framework:
- PyTorch

Training Platform:
- Google Colab GPU

Final Validation Accuracy:
- ~100%

---

## Repository Structure

```text
steel-surface-defect-classification/
│
├── data/
├── models/
│   └── best_model.pth
│
├── src/
│   ├── architecture.py
│   ├── config.py
│   ├── dataset.py
│   ├── engine.py
│   ├── predict.py
│   ├── resnet_model.py
│   └── visualize_dataset.py
│
├── experiments/
│   └── experiments.md
|
├── requirements.txt
├── README.md
└── LICENSE

