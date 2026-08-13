import pandas as pd

SRC = "data/cicids2017/cicids2017_cleaned.csv"
DST = "data/cicids2017/cicids2017_trimmed.csv"

print("Reading source file (this may take a minute)...")
df = pd.read_csv(SRC)

# Find the label column (name varies by dataset version)
label_col = "Attack Type"
print(f"Using label column: '{label_col}'")
print(df[label_col].value_counts())

# Keep all rows for rare attack classes, cap common classes (like BENIGN) at 20,000
CAP = 20000
parts = []
for label, group in df.groupby(label_col):
    if len(group) > CAP:
        group = group.sample(n=CAP, random_state=42)
    parts.append(group)

trimmed = pd.concat(parts).sample(frac=1, random_state=42)  # shuffle
trimmed.to_csv(DST, index=False)

print(f"\nDone. Trimmed dataset: {len(trimmed)} rows (from {len(df)})")
print(trimmed[label_col].value_counts())