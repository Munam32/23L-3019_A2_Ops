import json
import os
import sys
from pathlib import Path

import tensorflow as tf

MODEL_PATH = os.getenv("MODEL_PATH", "models/champion.keras")
META_PATH = os.getenv("MODEL_META_PATH", str(Path(MODEL_PATH).with_suffix(".json")))


class Classifier:
    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        self.model = tf.keras.models.load_model(model_path)
        meta = json.loads(Path(meta_path).read_text())
        self.threshold = float(os.getenv("THRESHOLD") or meta["threshold"])
        self.size = self.model.input_shape[1]
        self.name = f"visionguard-v{meta['version']}-{meta['run_name']}"

    def predict(self, image_bytes):
        img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, (self.size, self.size))
        p_ai = float(self.model(tf.cast(img, tf.float32)[None], training=False)[0, 0])
        is_ai = p_ai >= self.threshold
        return {"label": "AI-generated" if is_ai else "Real",
                "confidence": p_ai if is_ai else 1 - p_ai,
                "p_ai": p_ai, "threshold": self.threshold, "model": self.name}


if __name__ == "__main__":
    clf = Classifier()
    print(f"loaded {clf.name} | input {clf.size}x{clf.size} | threshold {clf.threshold}")
    for path in sys.argv[1:]:
        print(path, "->", clf.predict(Path(path).read_bytes()))
