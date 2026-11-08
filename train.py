import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from models import build_baseline, build_transfer, unfreeze_last_block
from pipeline import SEED, make_ds

tf.keras.utils.set_random_seed(SEED)
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)


PATIENCE = 5


def callbacks(name):
    return [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(f"models/{name}.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]


def plot(history, name):
    h = pd.DataFrame(history.history)
    h.index += 1
    h.to_csv(f"reports/{name}_history.csv", index_label="epoch")
    best = int(h["val_loss"].idxmin())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(h.index, h[key], label="train")
        ax.plot(h.index, h[f"val_{key}"], label="val")
        ax.axvline(best, color="red", ls="--", label=f"best epoch = {best}")
        ax.set_title(f"{name}: {key}")
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"reports/{name}_curves.png", dpi=130)
    plt.close()
    print(f"[{name}] best epoch = {best} | val_loss = {h.loc[best, 'val_loss']:.4f} | val_acc = {h.loc[best, 'val_accuracy']:.4f}")


def fit(model, name, epochs, train, val):
    history = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks(name), verbose=2)
    plot(history, name)
    return history


def trainable_names(model):
    base = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if base is None:
        return [l.name for l in model.layers if l.trainable and l.weights]
    return [f"{base.name}/{l.name}" for l in base.layers if l.trainable and l.weights] + [
        l.name for l in model.layers if l is not base and l.trainable and l.weights
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "mobilenet"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()
    PATIENCE = args.patience

    train = make_ds("train", training=True)
    val = make_ds("val", training=False)

    if args.model == "baseline":
        fit(build_baseline(), "baseline", args.epochs, train, val)
    else:
        model = build_transfer()
        print("STAGE 1 trainable layers:", trainable_names(model))
        fit(model, "mobilenet_stage1", args.epochs, train, val)
        unfreeze_last_block(model, lr=1e-5)
        names = trainable_names(model)
        print("STAGE 2 trainable layers:", names)
        Path("reports/mobilenet_trainable_layers.txt").write_text(
            "STAGE 2 trainable layers:\n" + "\n".join(names)
        )
        fit(model, "mobilenet_stage2", args.epochs, train, val)
