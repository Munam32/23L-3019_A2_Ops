import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from evalutil import pick_threshold
from models import build_transfer, set_trainable_tail
from pipeline import make_ds

ap = argparse.ArgumentParser()
ap.add_argument("--run-name", required=True)
ap.add_argument("--logging", choices=["manual", "autolog"], default="manual")
ap.add_argument("--backbone", choices=["mobilenetv2", "efficientnetb0"], default="mobilenetv2")
ap.add_argument("--lr", type=float, default=1.7e-4)
ap.add_argument("--dropout", type=float, default=0.2)
ap.add_argument("--unfrozen-layers", type=int, default=20)
ap.add_argument("--img-size", type=int, default=224)
ap.add_argument("--no-augment", dest="augment", action="store_false")
ap.add_argument("--max-epochs", type=int, default=30)
ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

load_dotenv(override=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")
mlflow.set_experiment("visionguard")
tf.keras.utils.set_random_seed(args.seed)


def sh(*cmd):
    return subprocess.check_output(cmd, text=True).strip()


git_commit = sh("git", "rev-parse", "--short", "HEAD")
git_dirty = bool(sh("git", "status", "--porcelain", "--", "*.py", "splits", "data.dvc"))
dvc_md5 = yaml.safe_load(Path("data.dvc").read_text())["outs"][0]["md5"]


class LogEpochs(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


if args.logging == "autolog":
    mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

Path("models").mkdir(exist_ok=True)
with mlflow.start_run(run_name=args.run_name):
    mlflow.set_tags({"git_commit": git_commit, "git_dirty": str(git_dirty), "dvc_data_md5": dvc_md5,
                     "seed": str(args.seed), "logging_mode": args.logging})
    mlflow.log_params({k: v for k, v in vars(args).items() if k not in ("run_name", "logging")})

    train = make_ds("train", True, img_size=args.img_size, augmentation=args.augment)
    val = make_ds("val", False, img_size=args.img_size)
    test = make_ds("test", False, img_size=args.img_size)

    model = build_transfer(args.dropout, args.lr, args.img_size, args.backbone)
    set_trainable_tail(model, args.unfrozen_layers, args.lr)
    mlflow.log_param("trainable_weight_tensors", len(model.trainable_weights))

    ckpt = f"models/{args.run_name}.keras"
    cbs = [tf.keras.callbacks.EarlyStopping("val_loss", patience=args.patience, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-7),
           tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True)]
    if args.logging == "manual":
        cbs.append(LogEpochs())
    hist = model.fit(train, validation_data=val, epochs=args.max_epochs, callbacks=cbs, verbose=2)

    h = pd.DataFrame(hist.history)
    best = int(h["val_loss"].idxmin()) + 1
    mlflow.log_metrics({"best_epoch": best, "best_val_loss": float(h["val_loss"].min()),
                        "best_val_auc": float(h.loc[best - 1, "val_auc"])})

    val_y = pd.read_csv("splits/val.csv")["label"].values
    test_y = pd.read_csv("splits/test.csv")["label"].values
    thr = pick_threshold(val_y, model.predict(val, verbose=0).ravel())
    p = model.predict(test, verbose=0).ravel()
    pred = (p >= thr).astype(int)
    mlflow.log_metrics({"threshold": thr, "test_accuracy": accuracy_score(test_y, pred),
                        "test_precision": precision_score(test_y, pred), "test_recall": recall_score(test_y, pred),
                        "test_f1": f1_score(test_y, pred), "test_roc_auc": roc_auc_score(test_y, p)})

    cm = confusion_matrix(test_y, pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["real", "ai"])
    ax.set_yticks([0, 1], ["real", "ai"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(test_y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(test_y, p):.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend()
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)

    req = Path("requirements_run.txt")
    req.write_text(sh(sys.executable, "-m", "pip", "freeze"))
    mlflow.log_artifact(str(req))
    mlflow.log_artifacts("splits", artifact_path="splits")
    mlflow.keras.log_model(model, name="model")
    print(f"[{args.run_name}] best epoch {best} | thr {thr:.2f} | test acc {accuracy_score(test_y, pred):.4f} "
          f"| test auc {roc_auc_score(test_y, p):.4f}")
