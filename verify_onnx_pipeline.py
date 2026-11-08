import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, "vercel_api")
from inference import Classifier  # noqa: E402
from onnx_infer import OnnxClassifier  # noqa: E402

keras_clf = Classifier()
onnx_clf = OnnxClassifier("models/champion.onnx", "models/champion_onnx.json")

df = pd.read_csv("splits/test.csv")
diffs, flips = [], 0
for path in df["path"]:
    data = Path(path).read_bytes()
    a, b = keras_clf.predict(data), onnx_clf.predict(data)
    diffs.append(abs(a["p_ai"] - b["p_ai"]))
    flips += a["label"] != b["label"]
d = np.array(diffs)
print(f"images: {len(d)} | max |dp|: {d.max():.2e} | mean |dp|: {d.mean():.2e} | label flips: {flips}")

a = tf.io.decode_image(tf.io.read_file(df["path"][0]), channels=3, expand_animations=False)
tf_r = tf.image.resize(a, (224, 224)).numpy()
from onnx_infer import resize_bilinear  # noqa: E402
print("resize max abs pixel diff vs tf.image.resize:", float(np.abs(tf_r - resize_bilinear(a.numpy().astype(np.float32), 224)).max()))
