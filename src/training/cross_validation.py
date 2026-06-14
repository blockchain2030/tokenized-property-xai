from sklearn.model_selection import StratifiedKFold

def five_fold_indices(labels, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(skf.split(range(len(labels)), labels))
