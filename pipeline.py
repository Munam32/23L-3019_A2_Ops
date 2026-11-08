import pandas as pd
import tensorflow as tf

SEED = 42
IMG_SIZE = 224
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.05, seed=SEED),
        tf.keras.layers.RandomZoom(0.1, seed=SEED),
        tf.keras.layers.RandomContrast(0.1, seed=SEED),
    ],
    name="augment",
)


def load(path, label, img_size=IMG_SIZE):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    return tf.cast(img, tf.float32), tf.cast(label, tf.float32)


def make_ds(split, training, batch=BATCH, img_size=IMG_SIZE, augmentation=True):
    df = pd.read_csv(f"splits/{split}.csv")
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: load(p, y, img_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch)
    if training and augmentation:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
