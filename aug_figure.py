import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import make_ds

plain = next(iter(make_ds("train", training=False)))
aug = next(iter(make_ds("train", training=True)))
print("batch shape:", aug[0].shape, aug[0].dtype, "| labels:", aug[1].shape)
print("pixel range:", float(aug[0].numpy().min()), float(aug[0].numpy().max()))

fig, axes = plt.subplots(2, 8, figsize=(18, 5))
for i in range(8):
    for r, (imgs, ys, tag) in enumerate([(plain[0], plain[1], "val-style"), (aug[0], aug[1], "augmented")]):
        axes[r, i].imshow(imgs[i].numpy().clip(0, 255).astype("uint8"))
        axes[r, i].set_title(f"{tag} y={int(ys[i])}", fontsize=8)
        axes[r, i].axis("off")
plt.tight_layout()
plt.savefig("reports/fig_augmented_batch.png", dpi=130)
print("saved reports/fig_augmented_batch.png")
