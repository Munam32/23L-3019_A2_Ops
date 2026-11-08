import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

df = pd.read_csv("reports/test_predictions_mobilenet_stage2.csv")
wrong = df[df["label"] != df["pred"]]
fn = wrong[wrong["label"] == 1].sort_values("prob_ai")
fp = wrong[wrong["label"] == 0].sort_values("prob_ai", ascending=False)
pick = pd.concat([fn.head(4), fp.head(10 - min(4, len(fn)))]).reset_index(drop=True)
pick.to_csv("reports/misclassified_10.csv", index=False)

fig, axes = plt.subplots(2, 5, figsize=(20, 9))
for ax, (i, r) in zip(axes.ravel(), pick.iterrows()):
    with Image.open(r["path"]) as im:
        w, h = im.size
        ax.imshow(im.convert("RGB"))
    kind = "FN (AI called real)" if r["label"] == 1 else "FP (real called AI)"
    ax.set_title(f"#{i + 1} {kind}\np(AI)={r['prob_ai']:.2f}  {w}x{h}\n{r['path'].split('/')[-1][:34]}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("reports/fig_misclassified.png", dpi=120)
print(pick[["path", "label", "prob_ai"]].to_string())
