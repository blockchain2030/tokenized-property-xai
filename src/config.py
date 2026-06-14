from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "outputs" / "models"

for p in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)
