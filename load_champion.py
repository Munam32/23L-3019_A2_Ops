import argparse
import json
from pathlib import Path

import mlflow
import tensorflow as tf
from dotenv import load_dotenv
from mlflow import MlflowClient

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--export", action="store_true", help="save champion model and threshold under models/")
args = ap.parse_args()

load_dotenv(dotenv_path=".env", override=True)
URI = "models:/visionguard@champion"
model = mlflow.keras.load_model(URI)
mv = MlflowClient().get_model_version_by_alias("visionguard", "champion")
threshold = float(mv.tags["threshold"])
size = model.input_shape[1]

img = tf.io.decode_image(tf.io.read_file(args.image), channels=3, expand_animations=False)
img = tf.image.resize(img, (size, size))[None]
p = float(model.predict(tf.cast(img, tf.float32), verbose=0)[0, 0])
print(f"model version {mv.version} (alias champion) | input {size}x{size} | threshold {threshold}")
print(f"p(AI) = {p:.4f} -> {'AI-generated' if p >= threshold else 'Real'}")

if args.export:
    Path("models").mkdir(exist_ok=True)
    model.save("models/champion.keras")
    Path("models/champion.json").write_text(json.dumps(
        {"version": mv.version, "threshold": threshold, "img_size": size, "run_name": mv.tags["run_name"]}))
    print("exported models/champion.keras and models/champion.json")
