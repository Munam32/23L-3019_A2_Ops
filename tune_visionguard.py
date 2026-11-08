import keras_tuner as kt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from models import build_transfer, compile_model
from pipeline import SEED, make_ds

tf.keras.utils.set_random_seed(SEED)


def build(hp):
    model = build_transfer(dropout=hp.Float("dropout", 0.2, 0.5, step=0.1))
    n = hp.Choice("unfrozen_layers", [0, 4, 10, 20])
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    if n:
        base.trainable = True
        for l in base.layers:
            l.trainable = False
        for l in [l for l in base.layers if not isinstance(l, layers.BatchNormalization)][-n:]:
            l.trainable = True
    compile_model(model, hp.Float("lr", 1e-5, 1e-3, sampling="log"))
    return model


train = make_ds("train", training=True)
val = make_ds("val", training=False)

tuner = kt.RandomSearch(build, objective=kt.Objective("val_loss", "min"), max_trials=8, seed=SEED,
                        directory="tuner_runs", project_name="visionguard", overwrite=True)
tuner.search(train, validation_data=val, epochs=10, verbose=2,
             callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=3, restore_best_weights=True)])

rows = []
for t in tuner.oracle.get_best_trials(8):
    rows.append({"trial": t.trial_id, **t.hyperparameters.values,
                 "val_loss": round(t.score, 4),
                 "val_auc": round(t.metrics.get_best_value("val_auc"), 4)})
table = pd.DataFrame(rows)
table.to_csv("reports/tuner_visionguard.csv", index=False)
print(table.to_string(index=False))
print("BEST:", tuner.get_best_hyperparameters(1)[0].values)
for col in ["unfrozen_layers", "dropout"]:
    print(f"\nmean val_loss by {col}:\n", table.groupby(col)["val_loss"].mean().round(4).to_string())
print("\ncorr(log10 lr, val_loss):", round(pd.Series(table["lr"]).apply(lambda v: __import__('math').log10(v)).corr(table["val_loss"]), 3))
