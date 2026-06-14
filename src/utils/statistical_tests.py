from scipy.stats import ttest_rel

def paired_t_test(model_a_scores, model_b_scores):
    stat, p = ttest_rel(model_a_scores, model_b_scores)
    return {'t_statistic': float(stat), 'p_value': float(p), 'significant_p_lt_0_05': bool(p < 0.05)}
