# Multimodal XAI Framework for Tokenized Property Investment Decisions

This repository is structured to validate the manuscript methodology and reported results for a multimodal explainable AI framework using four modality-specific encoders:

- Numerical encoder: MLP
- Text encoder: BERT
- Image encoder: ResNet-50 with Grad-CAM
- Blockchain encoder: Graph Attention Network (GAT)
- Fusion: 4-head cross-attention
- XAI: SHAP, LIME, Grad-CAM, Graph Attention visualization

## Reproducibility status

This repository supports two modes:

1. **Validation mode**: verifies that generated/reporting tables match the manuscript-reported values.
2. **Training mode**: trains the full PyTorch architecture when the real multimodal dataset is available.

The real dataset should be placed as follows:

```text
data/numerical/numerical_features.csv
data/textual/text_documents.csv
data/images/<property_id>/*.jpg
data/blockchain_graphs/<property_id>.pt
```

## Quick validation

```bash
pip install -r requirements.txt
python validate_manuscript_results.py
```

Expected output files:

```text
results/table_1_unimodal_results.csv
results/table_2_fusion_results.csv
results/table_3_xai_quality.csv
results/cross_validation_results.csv
results/statistical_tests.csv
```

## Important note

Old GradientBoosting `.joblib` models and proxy explanation plots should be placed in `legacy_baseline/`. They are not the final manuscript implementation.
