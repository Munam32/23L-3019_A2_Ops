import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

MODEL = "models/champion.keras"
meta = json.loads(Path("models/champion.json").read_text())
threshold = float(meta["threshold"])
model = tf.keras.models.load_model(MODEL)
size = model.input_shape[1]

base_idx = next(i for i, l in enumerate(model.layers) if isinstance(l, tf.keras.Model))
base = model.layers[base_idx]
pre_layers = model.layers[1:base_idx]
post_layers = model.layers[base_idx + 1:]
conv_layer = next(l for l in reversed(base.layers) if len(l.output.shape) == 4)
conv_model = tf.keras.Model(base.inputs, [conv_layer.output, base.output])
print(f"backbone: {base.name} | Grad-CAM layer: {conv_layer.name} {tuple(conv_layer.output.shape[1:])} | threshold {threshold}")


def load(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return tf.cast(tf.image.resize(img, (size, size)), tf.float32)


def forward(x):
    for layer in pre_layers:
        x = layer(x)
    conv_out, feats = conv_model(x)
    for layer in post_layers:
        feats = layer(feats, training=False) if layer.__class__.__name__ == "Dropout" else layer(feats)
    return conv_out, feats


def gradcam(x):
    with tf.GradientTape() as tape:
        conv_out, p = forward(x)
        tape.watch(conv_out)
        score = p[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam, float(p[0, 0])


df = pd.read_csv("splits/test.csv")
probs = np.array([float(forward(load(p)[None])[1][0, 0]) for p in df["path"]])
df["p_ai"] = probs
df["pred"] = (probs >= threshold).astype(int)
ok, bad = df[df.label == df.pred], df[df.label != df.pred]
picks = pd.concat([
    ok[ok.label == 1].nlargest(2, "p_ai").assign(group="correct"),
    ok[ok.label == 0].nsmallest(2, "p_ai").assign(group="correct"),
    bad[bad.label == 0].nlargest(2, "p_ai").assign(group="wrong"),
    bad[bad.label == 1].nsmallest(2, "p_ai").assign(group="wrong"),
]).reset_index(drop=True)
picks.to_csv("reports/gradcam_selection.csv", index=False)
print(f"test accuracy at threshold: {(df.label == df.pred).mean():.3f} | errors available: {len(bad)}")
print(picks[["group", "label", "p_ai", "path"]].to_string())

fig, axes = plt.subplots(4, 4, figsize=(16, 15))
for col, (_, r) in enumerate(picks.iterrows()):
    base_row = 0 if r["group"] == "correct" else 2
    col_i = col % 4
    img = load(r["path"])
    cam, p = gradcam(img[None])
    heat = tf.image.resize(cam[..., None], (size, size)).numpy()[..., 0]
    rgb = img.numpy().astype("uint8")
    overlay = (0.55 * rgb + 0.45 * plt.cm.jet(heat)[..., :3] * 255).astype("uint8")
    truth = "AI" if r["label"] else "real"
    pred = "AI" if p >= threshold else "real"
    title = f"{r['group'].upper()}: truth {truth}, predicted {pred}\np(AI)={p:.2f}  {Path(r['path']).name[:28]}"
    axes[base_row, col_i].imshow(rgb)
    axes[base_row, col_i].set_title(title, fontsize=9)
    axes[base_row + 1, col_i].imshow(overlay)
    axes[base_row + 1, col_i].set_title("Grad-CAM (red = pushes toward AI)", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.savefig("reports/fig_gradcam.png", dpi=120)
print("saved reports/fig_gradcam.png and reports/gradcam_selection.csv")
