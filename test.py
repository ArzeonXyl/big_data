# test_model_load.py

import pickle
from pathlib import Path

print("=== MODEL LOAD TEST ===")

# 1. Deteksi base folder (tanpa __file__ fallback)
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # kalau jalan dari Jupyter
    BASE_DIR = Path().resolve()

print(f"Base dir: {BASE_DIR}")

# 2. Build path ke model
model_path = BASE_DIR / "data" / "model_sarimax_transaksi.pkl"
print(f"Model path: {model_path}")
print(f"Exists: {model_path.exists()}")

# 3. Coba load
try:
    with open(model_path, "rb") as f:
        model_sarimax = pickle.load(f)

    print("✅ Model loaded successfully!")
    print(f"Type: {type(model_sarimax)}")

except Exception as e:
    print("❌ Failed to load model")
    print(f"Error: {e}")
