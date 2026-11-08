import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv(override=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow_toy.db")
mlflow.set_experiment("toy_fashion_mnist")

tf.keras.utils.set_random_seed(42)
(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
x, y = x[:10000] / 255.0, y[:10000]


class LogEpochs(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)


for lr in (1e-3, 1e-2):
    with mlflow.start_run(run_name=f"toy_lr_{lr}"):
        params = {"lr": lr, "units": 128, "epochs": 5, "seed": 42}
        mlflow.log_params(params)
        mlflow.set_tags({"stage": "toy", "dataset": "fashion_mnist_10k"})
        model = tf.keras.Sequential([
            tf.keras.layers.Input((28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(params["units"], activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        h = model.fit(x, y, epochs=params["epochs"], validation_split=0.2, verbose=0, callbacks=[LogEpochs()])
        fig = plt.figure()
        plt.plot(h.history["loss"], label="train")
        plt.plot(h.history["val_loss"], label="val")
        plt.xlabel("epoch")
        plt.title(f"loss, lr={lr}")
        plt.legend()
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)
        mlflow.keras.log_model(model, "model")
        print("run done, lr =", lr, "val_acc =", round(h.history["val_accuracy"][-1], 4))
