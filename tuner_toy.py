import keras_tuner as kt
import pandas as pd
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
x = x[:10000] / 255.0
y = y[:10000]


def build(hp):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=8, seed=42,
                        directory="tuner_runs", project_name="toy_fashion", overwrite=True)
tuner.search(x, y, epochs=5, validation_split=0.2, verbose=0)

rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": round(t.score, 4)}
        for t in tuner.oracle.get_best_trials(8)]
table = pd.DataFrame(rows)
table.to_csv("reports/tuner_toy.csv", index=False)
print(table.to_string(index=False))
print("BEST:", tuner.get_best_hyperparameters(1)[0].values)
