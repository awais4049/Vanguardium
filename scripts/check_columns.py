import pandas as pd

df = pd.read_csv("data/cicids2017/cicids2017_cleaned.csv", nrows=5)
print("Columns found:")
for c in df.columns:
    print(repr(c))