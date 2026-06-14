import argparse
from pathlib import Path
import joblib
from src.config import OUTPUT_DIR, MODEL_DIR
from src.data_loader import load_master_dataset
from src.experiments import run_unimodal_baselines, run_fusion_comparison, run_cross_validation, paper_reported_tables
from src.explainability import (
    shap_like_feature_importance, lime_like_local_explanation, gradcam_metadata_proxy,
    graph_attention_proxy, explainability_quality_table,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to benchmark Excel workbook")
    args = parser.parse_args()

    df = load_master_dataset(args.data)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    unimodal, models = run_unimodal_baselines(df)
    fusion = run_fusion_comparison(df)
    cv = run_cross_validation(df)
    xai_quality = explainability_quality_table()
    paper_t1, paper_t2, paper_t3 = paper_reported_tables()

    unimodal.to_csv(OUTPUT_DIR / "unimodal_results.csv", index=False)
    fusion.to_csv(OUTPUT_DIR / "fusion_results.csv", index=False)
    cv.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)
    xai_quality.to_csv(OUTPUT_DIR / "explainability_quality.csv", index=False)
    paper_t1.to_csv(OUTPUT_DIR / "paper_reported_table_1.csv", index=False)
    paper_t2.to_csv(OUTPUT_DIR / "paper_reported_table_2.csv", index=False)
    paper_t3.to_csv(OUTPUT_DIR / "paper_reported_table_3.csv", index=False)

    for name, model in models.items():
        joblib.dump(model, MODEL_DIR / f"{name.replace(' ', '_').replace('/', '_')}.joblib")

    shap_imp = shap_like_feature_importance(df)
    lime_exp = lime_like_local_explanation(df)
    gradcam_proxy = gradcam_metadata_proxy(df)
    graph_proxy = graph_attention_proxy(df)

    shap_imp.to_csv(OUTPUT_DIR / "shap_style_feature_importance.csv", index=False)
    lime_exp.to_csv(OUTPUT_DIR / "lime_style_local_explanation.csv", index=False)
    gradcam_proxy.to_csv(OUTPUT_DIR / "gradcam_metadata_proxy.csv", index=False)
    graph_proxy.to_csv(OUTPUT_DIR / "graph_attention_proxy.csv", index=False)

    print("\nMeasured unimodal/multimodal results:")
    print(unimodal.to_string(index=False))
    print("\nFusion comparison:")
    print(fusion.to_string(index=False))
    print("\nOutputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
