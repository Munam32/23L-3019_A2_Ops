import io

import pytest
from PIL import Image

from api import create_api


class FakeClassifier:
    name = "fake-v0"
    threshold = 0.5

    def predict(self, data):
        return {"label": "Real", "confidence": 0.8, "p_ai": 0.2, "threshold": 0.5, "model": self.name}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.2")
    return create_api(classifier=FakeClassifier()).test_client()


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "blue").save(buf, "PNG")
    return buf.getvalue()


def post(client, data, name="a.png", key="test-key"):
    headers = {"X-API-Key": key} if key else {}
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, headers=headers,
                       content_type="multipart/form-data")


def test_health_is_public(client):
    assert client.get("/health").get_json()["status"] == "ok"


def test_predict_with_key(client):
    r = post(client, png_bytes())
    assert r.status_code == 200
    body = r.get_json()
    assert body["label"] == "Real" and body["filename"] == "a.png"


def test_missing_or_wrong_key_rejected(client):
    assert post(client, png_bytes(), key=None).status_code == 401
    assert post(client, png_bytes(), key="wrong").status_code == 401


def test_bad_uploads_rejected(client):
    assert post(client, b"not an image", "x.jpg").status_code == 400
    assert post(client, b"GIF89a", "x.gif").status_code == 400
    assert post(client, png_bytes() + b"0" * 300_000).status_code == 413


def test_refuses_to_start_without_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        create_api(classifier=FakeClassifier())
