# tokenized-property-xai
Multimodal Explainable AI Framework for Tokenized Property Investment Evaluation
Multimodal Explainable AI for Tokenized Property Investment
GitHub-ready reproduction package for the research  “Multimodal Explainable AI Framework for Evaluating Tokenized Property Investment Decisions.”
This repository uses the supplied benchmark workbook:
`data/Multimodal_Tokenized_Property_Benchmark_Dataset_1250.xlsx`
Important note
Because the workbook contains visual metadata and textual metadata rather than raw property images, full documents, and real blockchain graphs, this repository includes two layers:
Executable benchmark layer — runs on the supplied Excel file and reproduces the paper-style metric tables using available numerical, textual, visual-metadata, and blockchain-metadata columns.
Full architecture layer — code structure showing where BERT, ResNet-50, GAT, SHAP, LIME, Grad-CAM, and graph-attention modules should be connected when raw documents, images, and blockchain graph files are available.
Quick start in Google Colab
```bash
!git clone <your-repo-url>
%cd tokenized-property-xai
!pip install -r requirements.txt
!python run_all.py --data data/Multimodal_Tokenized_Property_Benchmark_Dataset_1250.xlsx
```
Recommended runtime: Python 3 + T4 GPU. The executable benchmark uses scikit-learn and can also run on CPU.
Main outputs
After running, check:
`outputs/unimodal_results.csv`
`outputs/fusion_results.csv`
`outputs/cross_validation_results.csv`
`outputs/explainability_quality.csv`
`figures/shap_feature_importance.png`
`figures/lime_like_local_explanation.png`
`figures/gradcam_metadata_proxy.png`
`figures/graph_attention_proxy.png`
Repository structure
```text
.
├── data/
│   └── Multimodal_Tokenized_Property_Benchmark_Dataset_1250.xlsx
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_groups.py
│   ├── models_sklearn.py
│   ├── experiments.py
│   ├── explainability.py
│   └── full_architecture_placeholders.py
├── outputs/
├── figures/
├── run_all.py
├── requirements.txt
└── README.md
```
How this maps 
Paper component	Executable implementation in this repo
Numerical encoder / MLP	Numerical feature pipeline + tree/boosting classifier
Textual encoder / BERT	Textual metadata + categorical/text-risk features
Visual encoder / ResNet-50	Visual metadata proxy; raw image hook included
Blockchain encoder / GAT	Blockchain graph-stat metadata proxy; graph hook included
Cross-attention fusion	Strong multimodal fusion classifier using all modality groups
SHAP	Permutation/feature attribution plot; SHAP hook included
LIME	Local perturbation-style explanation plot
Grad-CAM	Visual metadata proxy plot; raw-image Grad-CAM hook included
Graph attention	Wallet/graph feature importance proxy; GAT hook included
Expected benchmark behavior
The benchmark dataset is designed so the multimodal model should outperform individual modalities. Exact values can differ slightly by library version and random seed, but the intended result pattern is:
multimodal model highest accuracy/F1/ROC-AUC,
blockchain-only strongest unimodal baseline,
early/late fusion below the stronger multimodal fusion,
explainability artifacts generated for all four modality groups.
