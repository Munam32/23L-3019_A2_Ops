import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from pipeline import make_ds

MODELS = {"baseline": "models/baseline.keras", "mobilenet_stage2": "models/mobilenet_stage2.keras"}
val_y = pd.read_csv("splits/val.csv")["label"].values
test_df = pd.read_csv("splits/test.csv")
test_y = test_df["label"].values


def predict(model, split):
    return model.predict(make_ds(split, training=False), verbose=0).ravel()


def pick_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y, p >= t, beta=2) for t in grid]
    return float(grid[int(np.argmax(scores))])


rows, roc_data, cms = [], {}, {}
for name, path in MODELS.items():
    model = tf.keras.models.load_model(path)
    t = pick_threshold(val_y, predict(model, "val"))
    p = predict(model, "test")
    pred = (p >= t).astype(int)
    rows.append({"model": name, "threshold": t, "accuracy": accuracy_score(test_y, pred),
                 "precision": precision_score(test_y, pred), "recall": recall_score(test_y, pred),
                 "f1": f1_score(test_y, pred), "roc_auc": roc_auc_score(test_y, p)})
    roc_data[name] = roc_curve(test_y, p)
    cms[name] = confusion_matrix(test_y, pred)
    out = test_df.copy()
    out["prob_ai"], out["pred"] = p, pred
    out.to_csv(f"reports/test_predictions_{name}.csv", index=False)

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv("reports/test_metrics.csv", index=False)
print(metrics.to_string(index=False))
for n, cm in cms.items():
    print(n, "confusion [[TN FP][FN TP]] (0=real, 1=ai):", cm.tolist())

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (n, cm) in zip(axes, cms.items()):
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["real", "ai"])
    ax.set_yticks([0, 1], ["real", "ai"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(n)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("reports/fig_confusion.png", dpi=130)
plt.close()

plt.figure(figsize=(5.5, 5))
for n, (fpr, tpr, _) in roc_data.items():
    auc = metrics.loc[metrics.model == n, "roc_auc"].item()
    plt.plot(fpr, tpr, label=f"{n} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="chance")
plt.xlabel("false positive rate (real flagged as AI)")
plt.ylabel("true positive rate (AI caught)")
plt.title("ROC curve, test set")
plt.legend()
plt.tight_layout()
plt.savefig("reports/fig_roc.png", dpi=130)
