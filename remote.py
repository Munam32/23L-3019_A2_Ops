import logging
import os

import requests

log = logging.getLogger(__name__)


class ServiceUnavailable(Exception):
    pass


class RemoteClassifier:
    """Same interface as inference.Classifier, but predictions come from the hosted model API."""

    def __init__(self, url=None, api_key=None, timeout=60):
        self.url = (url or os.getenv("MODEL_API_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("MODEL_API_KEY", "")
        if not self.url or not self.api_key:
            raise RuntimeError("MODEL_API_URL and MODEL_API_KEY must be set for the remote backend.")
        self.timeout = timeout
        self.name, self.threshold = "remote-model", None
        try:
            info = requests.get(f"{self.url}/health", timeout=self.timeout).json()
            self.name, self.threshold = info["model"], info["threshold"]
        except (requests.RequestException, ValueError, KeyError):
            log.warning("Model API health check failed at start-up; will retry on first prediction.")

    def predict(self, image_bytes):
        filename = "upload.png" if image_bytes.startswith(b"\x89PNG") else "upload.jpg"
        try:
            r = requests.post(f"{self.url}/predict", headers={"X-API-Key": self.api_key},
                              files={"image": (filename, image_bytes)}, timeout=self.timeout)
        except requests.RequestException as err:
            raise ServiceUnavailable(f"model API unreachable: {err}") from err
        if r.status_code == 400:
            raise ValueError(r.json().get("error", "model API rejected the image"))
        if r.status_code != 200:
            raise ServiceUnavailable(f"model API returned HTTP {r.status_code}")
        result = r.json()
        self.name, self.threshold = result["model"], result["threshold"]
        return {k: result[k] for k in ("label", "confidence", "p_ai", "threshold", "model")}
