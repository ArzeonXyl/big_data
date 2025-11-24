import pandas as pd

# Load CSV asli
df = pd.read_csv("data/data_segmentasi.csv", parse_dates=['trans_datetime'])

# Simpan versi pickle (lebih cepat load)
df.to_pickle("data/data_segmentasi.pkl")
print("Pickle created!")
