import hashlib
import os
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path("data")
CLASSES = {"AiArtData": "ai", "RealArt": "real"}
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

rows = []
for folder, label in CLASSES.items():
    for p in sorted((ROOT / folder).rglob("*")):
        if not p.is_file():
            continue
        row = {"path": p.as_posix(), "class": label, "file_size": p.stat().st_size,
               "width": None, "height": None, "mode": None, "format": None,
               "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "corrupt": False}
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
        except Exception:
            row["corrupt"] = True
        rows.append(row)

df = pd.DataFrame(rows)
df["duplicate"] = df.duplicated("sha256", keep="first")
df.to_csv(OUT / "audit.csv", index=False)

pd.set_option("display.width", 200)
print("Counts per class:\n", df["class"].value_counts(), "\n")
ok = df[~df["corrupt"]]
print("min/median/max per class:")
print(ok.groupby("class")[["width", "height", "file_size"]].agg(["min", "median", "max"]).T, "\n")
print("Corrupt files:", int(df["corrupt"].sum()))
print(df[df["corrupt"]]["path"].to_string(index=False))
print("Exact duplicates (extra copies):", int(df["duplicate"].sum()))
dups = df[df.duplicated("sha256", keep=False)].sort_values("sha256")
print(dups[["sha256", "class", "path"]].to_string(index=False))
print("\nColour modes:\n", df["mode"].value_counts(dropna=False))
print("\nFormats:\n", df["format"].value_counts(dropna=False))
