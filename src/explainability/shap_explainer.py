"""SHAP analysis for numerical and blockchain-level embeddings/features."""
import shap

def make_shap_summary(model_predict_fn, background, samples, feature_names, output_path):
    explainer = shap.KernelExplainer(model_predict_fn, background)
    values = explainer.shap_values(samples)
    shap.summary_plot(values, samples, feature_names=feature_names, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return values
