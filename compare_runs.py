import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv(dotenv_path=".env", override=True)
c = MlflowClient()
exp = c.get_experiment_by_name("visionguard")
CHANGE = {"ref_manual": "reference", "ref_autolog": "reference (autolog)", "lr_1e-3": "lr 1.7e-4 -> 1e-3",
          "no_augmentation": "augmentation off", "frozen_backbone": "unfrozen layers 20 -> 0",
          "img_size_160": "image size 224 -> 160", "efficientnetb0": "backbone MobileNetV2 -> EfficientNetB0"}

rows = {}
for r in c.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"]):
    n = r.info.run_name
    if n not in CHANGE or r.data.tags.get("git_dirty") != "False":
        continue
    m = r.data.metrics
    rows[n] = {"run": n, "change": CHANGE[n], "best_epoch": int(m["best_epoch"]),
               "val_loss": m["best_val_loss"], "val_auc": m["best_val_auc"], "threshold": m["threshold"],
               "test_auc": m["test_roc_auc"], "test_acc": m["test_accuracy"], "test_f1": m["test_f1"],
               "test_recall": m["test_recall"], "test_precision": m["test_precision"]}
df = pd.DataFrame(rows.values()).round(4)
df.to_csv("reports/run_comparison.csv", index=False)
pd.set_option("display.width", 250)
print(df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
order = df.sort_values("val_loss")
axes[0].barh(order["run"], order["val_loss"], color="tab:orange")
axes[0].set_xlabel("best validation loss (lower is better)")
axes[1].barh(order["run"], order["test_auc"], color="tab:blue")
axes[1].set_xlabel("test ROC-AUC (higher is better)")
axes[1].set_xlim(0.5, 1.0)
for ax in axes:
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig("reports/fig_run_comparison.png", dpi=130)
