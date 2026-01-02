from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

UNIFIC_DIR = BASE_DIR / "data" / "unificado"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUTA_DATASET = UNIFIC_DIR / "dataset_unificado_v2.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2