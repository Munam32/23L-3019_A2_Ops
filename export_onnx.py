import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import tensorflow as tf

SRC = "models/champion.keras"
DST = "models/champion.onnx"

model = tf.keras.models.load_model(SRC)
size = model.input_shape[1]
model(np.zeros((1, size, size, 3), dtype="float32"), training=False)
SAVED = "models/_savedmodel"
model.export(SAVED, format="tf_saved_model")
subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--saved-model", SAVED, "--output", DST,
                "--opset", "17"], check=True)
shutil.rmtree(SAVED, ignore_errors=True)
print("exported", DST, round(Path(DST).stat().st_size / 1e6, 1), "MB")

sess = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
inputs = [i.name for i in sess.get_inputs()]
print("onnx inputs:", inputs, "output:", sess.get_outputs()[0].name)
assert len(inputs) == 1, "graph still has extra inputs"
in_name = inputs[0]

df = pd.read_csv("splits/test.csv").sample(40, random_state=1)
worst = 0.0
for path in df["path"]:
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    x = tf.cast(tf.image.resize(img, (size, size)), tf.float32)[None].numpy()
    keras_p = float(model(x, training=False)[0, 0])
    onnx_p = float(sess.run(None, {in_name: x})[0][0, 0])
    worst = max(worst, abs(keras_p - onnx_p))
print(f"max |keras - onnx| over {len(df)} test images: {worst:.2e}")

meta = json.loads(Path("models/champion.json").read_text())
Path("models/champion_onnx.json").write_text(json.dumps({**meta, "img_size": size}))
