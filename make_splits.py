from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
df = pd.read_csv("reports/audit.csv")
df = df[~df["corrupt"] & ~df["duplicate"]][["path", "class"]]
df["label"] = (df["class"] == "ai").astype(int)

train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)

out = Path("splits")
out.mkdir(exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(out / f"{name}.csv", index=False)
    print(name, len(part), part["class"].value_counts().to_dict())
