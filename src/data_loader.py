import pandas as pd
from .feature_groups import ALL_MODALITY_FEATURES, TARGET


def load_master_dataset(path: str) -> pd.DataFrame:
    """Load the Master_Dataset sheet and validate required columns."""
    df = pd.read_excel(path, sheet_name="Master_Dataset")
    missing = [c for c in ALL_MODALITY_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df
