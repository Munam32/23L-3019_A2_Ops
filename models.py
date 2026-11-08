import tensorflow as tf
from tensorflow.keras import layers

from pipeline import IMG_SIZE


def metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]


def build_baseline(dropout=0.4, lr=1e-3, img_size=IMG_SIZE):
    inp = layers.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="rescale")(inp)
    for i, f in enumerate([32, 64, 128, 128], start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    model = tf.keras.Model(inp, out, name="baseline_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=metrics(),
    )
    return model


def build_transfer(dropout=0.3, lr=1e-3, img_size=IMG_SIZE, backbone="mobilenetv2"):
    shape = (img_size, img_size, 3)
    inp = layers.Input(shape, name="image")
    if backbone == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape)
        x = layers.Rescaling(1 / 127.5, offset=-1, name="preprocess_mobilenetv2")(inp)
    elif backbone == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=shape)
        x = inp  # EfficientNetB0 rescales raw 0-255 pixels internally
    else:
        raise ValueError(backbone)
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    model = tf.keras.Model(inp, out, name="mobilenetv2_head")
    compile_model(model, lr)
    return model


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=metrics(),
    )


def set_trainable_tail(model, n, lr):
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    base.trainable = n > 0
    for l in base.layers:
        l.trainable = False
    if n:
        for l in [l for l in base.layers if not isinstance(l, layers.BatchNormalization)][-n:]:
            l.trainable = True
    compile_model(model, lr)


def unfreeze_last_block(model, lr=1e-5):
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    base.trainable = True
    for l in base.layers:
        in_last_block = l.name.startswith("block_16") or l.name.startswith("Conv_1")
        l.trainable = in_last_block and not isinstance(l, layers.BatchNormalization)
    compile_model(model, lr)
    return base


if __name__ == "__main__":
    import pandas as pd

    m = build_baseline()
    m.summary()
    rows = [(l.name, l.__class__.__name__, str(l.output.shape), l.count_params()) for l in m.layers]
    t = pd.DataFrame(rows, columns=["layer", "type", "output_shape", "params"])
    t.to_csv("reports/baseline_layers.csv", index=False)
    print(t.to_string(index=False))
