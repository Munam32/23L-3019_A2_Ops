import io
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def _axis(n_in, n_out):
    pos = (np.arange(n_out, dtype=np.float32) + 0.5) * (n_in / n_out) - 0.5
    floor = np.floor(pos)
    lower = np.maximum(floor, 0).astype(np.int64)
    upper = np.minimum(np.ceil(pos), n_in - 1).astype(np.int64)
    return lower, upper, (pos - floor).astype(np.float32)


def resize_bilinear(img, size):
    """Same arithmetic as tf.image.resize(method='bilinear', antialias=False) on HxWx3 float32."""
    h, w, _ = img.shape
    y0, y1, fy = _axis(h, size)
    x0, x1, fx = _axis(w, size)
    fy, fx = fy[:, None, None], fx[None, :, None]
    top = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    bottom = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return top * (1 - fy) + bottom * fy


class OnnxClassifier:
    def __init__(self, model_path=None, meta_path=None):
        here = Path(__file__).parent / "model"
        model_path = model_path or os.getenv("ONNX_MODEL_PATH") or str(here / "champion.onnx")
        meta_path = meta_path or os.getenv("ONNX_META_PATH") or str(here / "champion_onnx.json")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        meta = json.loads(Path(meta_path).read_text())
        self.threshold = float(os.getenv("THRESHOLD") or meta["threshold"])
        self.size = int(meta["img_size"])
        self.name = f"visionguard-v{meta['version']}-{meta['run_name']}-onnx"

    def predict(self, image_bytes):
        with Image.open(io.BytesIO(image_bytes)) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
        x = resize_bilinear(arr, self.size)[None].astype(np.float32)
        p_ai = float(self.session.run(None, {self.input_name: x})[0][0, 0])
        is_ai = p_ai >= self.threshold
        return {"label": "AI-generated" if is_ai else "Real",
                "confidence": p_ai if is_ai else 1 - p_ai,
                "p_ai": p_ai, "threshold": self.threshold, "model": self.name}
