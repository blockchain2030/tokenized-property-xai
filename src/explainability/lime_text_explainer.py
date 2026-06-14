"""LIME text explanation for BERT text classifier path."""
from lime.lime_text import LimeTextExplainer

RISK_LABELS = ['unsuitable', 'suitable']

def explain_text_instance(text, predict_proba_fn, output_html='figures/lime_text_explanation.html'):
    explainer = LimeTextExplainer(class_names=RISK_LABELS)
    exp = explainer.explain_instance(text, predict_proba_fn, num_features=10)
    exp.save_to_file(output_html)
    return exp.as_list()
