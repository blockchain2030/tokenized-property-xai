# Multimodal Explainable AI for Tokenized Property Investment

GitHub-ready reproduction package for the research  **“Multimodal Explainable AI Framework for Evaluating Tokenized Property Investment Decisions.”**

This repository uses the supplied benchmark workbook:

`data/Multimodal\\\_Tokenized\\\_Property\\\_Benchmark\\\_Dataset\\\_1250.xlsx`

## Important note

Because the workbook contains visual metadata and textual metadata rather than raw property images, full documents, and real blockchain graphs, this repository includes two layers:

1. **Executable benchmark layer** — runs on the supplied Excel file and reproduces the paper-style metric tables using available numerical, textual, visual-metadata, and blockchain-metadata columns.
2. **Full architecture layer** — code structure showing where BERT, ResNet-50, GAT, SHAP, LIME, Grad-CAM, and graph-attention modules should be connected when raw documents, images, and blockchain graph files are available.

## Quick start in Google Colab

```bash
!git clone <your-repo-url>
%cd tokenized-property-xai
!pip install -r requirements.txt
!python run\\\_all.py --data data/Multimodal\\\_Tokenized\\\_Property\\\_Benchmark\\\_Dataset\\\_1250.xlsx
```

Recommended runtime: **Python 3 + T4 GPU**. The executable benchmark uses scikit-learn and can also run on CPU.

## Main outputs

After running, check:

* `outputs/unimodal\\\_results.csv`
* `outputs/fusion\\\_results.csv`
* `outputs/cross\\\_validation\\\_results.csv`
* `outputs/explainability\\\_quality.csv`
* `figures/shap\\\_feature\\\_importance.png`
* `figures/lime\\\_like\\\_local\\\_explanation.png`
* `figures/gradcam\\\_metadata\\\_proxy.png`
* `figures/graph\\\_attention\\\_proxy.png`

## Repository structure

```text
.
├── data/
│   └── Multimodal\\\_Tokenized\\\_Property\\\_Benchmark\\\_Dataset\\\_1250.xlsx
├── src/
│   ├── config.py
│   ├── data\\\_loader.py
│   ├── feature\\\_groups.py
│   ├── models\\\_sklearn.py
│   ├── experiments.py
│   ├── explainability.py
│   └── full\\\_architecture\\\_placeholders.py
├── outputs/
├── figures/
├── run\\\_all.py
├── requirements.txt
└── README.md
```

## How this maps

|Paper component|Executable implementation in this repo|
|-|-|
|Numerical encoder / MLP|Numerical feature pipeline + tree/boosting classifier|
|Textual encoder / BERT|Textual metadata + categorical/text-risk features|
|Visual encoder / ResNet-50|Visual metadata proxy; raw image hook included|
|Blockchain encoder / GAT|Blockchain graph-stat metadata proxy; graph hook included|
|Cross-attention fusion|Strong multimodal fusion classifier using all modality groups|
|SHAP|Permutation/feature attribution plot; SHAP hook included|
|LIME|Local perturbation-style explanation plot|
|Grad-CAM|Visual metadata proxy plot; raw-image Grad-CAM hook included|
|Graph attention|Wallet/graph feature importance proxy; GAT hook included|

## Expected benchmark behavior

The benchmark dataset is designed so the multimodal model should outperform individual modalities. Exact values can differ slightly by library version and random seed, but the intended result pattern is:

* multimodal model highest accuracy/F1/ROC-AUC,
* blockchain-only strongest unimodal baseline,
* early/late fusion below the stronger multimodal fusion,
* explainability artifacts generated for all four modality groups.

