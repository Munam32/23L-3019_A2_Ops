import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

df = pd.read_csv("reports/audit.csv")
df = df[~df["corrupt"] & ~df["duplicate"]]

fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
for r, label in enumerate(["ai", "real"]):
    sample = df[df["class"] == label].sample(6, random_state=42)
    for c, path in enumerate(sample["path"]):
        with Image.open(path) as im:
            axes[r, c].imshow(im.convert("RGB"))
        axes[r, c].axis("off")
        axes[r, c].set_title(f"{label}: {path.split('/')[-1]}", fontsize=8)
plt.tight_layout()
plt.savefig("reports/fig_samples.png", dpi=130)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, col, log in zip(axes, ["width", "height", "file_size"], [False, False, True]):
    for label, colour in [("ai", "tab:red"), ("real", "tab:blue")]:
        ax.hist(df[df["class"] == label][col], bins=40, alpha=0.6, label=label, color=colour, log=log)
    ax.set_title(f"{col} per class" + (" (log y)" if log else ""))
    ax.set_xlabel(col)
    ax.set_ylabel("images")
    ax.legend()
plt.tight_layout()
plt.savefig("reports/fig_hist.png", dpi=130)
print("saved reports/fig_samples.png, reports/fig_hist.png")
