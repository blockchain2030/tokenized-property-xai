# Manuscript Alignment Checklist

Use this checklist before journal submission.

## Methodology
- [ ] Numerical model is MLP [256, 128, 128], not GradientBoosting.
- [ ] Text model uses BERT and text documents, not tabular text metadata only.
- [ ] Image model uses ResNet-50 and actual property images.
- [ ] Blockchain model uses GAT and graph tensors.
- [ ] Fusion uses 4-head cross-attention.
- [ ] Decision layer contains three output heads.

## Results
- [ ] Table I generated from `results/table_1_unimodal_results.csv`.
- [ ] Table II generated from `results/table_2_fusion_results.csv`.
- [ ] Table III generated from `results/table_3_xai_quality.csv`.
- [ ] Cross-validation generated from `results/cross_validation_results.csv`.
- [ ] Statistical tests generated from `results/statistical_tests.csv`.

## Figures
- [ ] SHAP summary is based on model attribution.
- [ ] LIME figure highlights textual clauses.
- [ ] Grad-CAM shows property image heatmap overlay.
- [ ] Graph Attention shows wallet network topology with node attention.
