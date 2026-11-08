import io

import pytest
import requests
from PIL import Image

from app import create_app
from remote import RemoteClassifier, ServiceUnavailable


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body


API_RESULT = {"label": "AI-generated", "confidence": 0.91, "p_ai": 0.91, "threshold": 0.2,
              "model": "cloud-v2-onnx", "filename": "upload.png", "latency_ms": 100}


def png():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "green").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def fake_api(monkeypatch):
    calls = {"post": []}

    def get(url, **kw):
        return FakeResponse(200, {"model": "cloud-v2-onnx", "threshold": 0.2, "status": "ok"})

    def post(url, headers=None, files=None, timeout=None):
        calls["post"].append((url, headers, files))
        return calls.get("response", FakeResponse(200, API_RESULT))

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    return calls


def make_client(classifier):
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite://"}, classifier=classifier)
    return app.test_client()


def upload(client):
    return client.post("/predict", data={"image": (io.BytesIO(png()), "a.png")},
                       content_type="multipart/form-data")


def test_remote_classifier_sends_key_and_returns_result(fake_api):
    clf = RemoteClassifier("https://api.example", "secret")
    out = clf.predict(png())
    url, headers, files = fake_api["post"][0]
    assert url == "https://api.example/predict" and headers == {"X-API-Key": "secret"}
    assert files["image"][0] == "upload.png"
    assert out["label"] == "AI-generated" and clf.name == "cloud-v2-onnx"


def test_app_uses_remote_result_and_logs_it(fake_api):
    client = make_client(RemoteClassifier("https://api.example", "secret"))
    r = upload(client)
    assert r.status_code == 200 and b"AI-generated" in r.data
    assert b"cloud-v2-onnx" in client.get("/history").data


def test_remote_failure_shows_502_and_is_not_logged(fake_api):
    fake_api["response"] = FakeResponse(500)
    client = make_client(RemoteClassifier("https://api.example", "secret"))
    r = upload(client)
    assert r.status_code == 502 and b"Model service unavailable" in r.data
    assert b"cloud-v2-onnx" not in client.get("/history").data


def test_remote_network_error_raises_service_unavailable(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(200, {"model": "m", "threshold": 0.2}))

    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(ServiceUnavailable):
        RemoteClassifier("https://api.example", "secret").predict(png())


def test_requires_url_and_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_URL", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        RemoteClassifier()
