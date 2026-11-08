import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vercel_api"))

from api import create_api  # noqa: E402
from restore_path import RestorePath  # noqa: E402


class FakeClassifier:
    name = "fake-v0"
    threshold = 0.5

    def predict(self, data):
        return {"label": "Real", "confidence": 0.8, "p_ai": 0.2, "threshold": 0.5, "model": self.name}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    app = create_api(classifier=FakeClassifier())
    app.wsgi_app = RestorePath(app.wsgi_app)
    return app.test_client()


def test_rewritten_health_reaches_health_route(client):
    r = client.get("/api/index?__path=health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_rewritten_predict_keeps_other_query_params_and_auth(client):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, "PNG")
    body = {"image": (io.BytesIO(buf.getvalue()), "a.png")}
    assert client.post("/api/index?__path=predict", data=body,
                       content_type="multipart/form-data").status_code == 401
    ok = client.post("/api/index?__path=predict", data={"image": (io.BytesIO(buf.getvalue()), "a.png")},
                     headers={"X-API-Key": "test-key"}, content_type="multipart/form-data")
    assert ok.status_code == 200 and ok.get_json()["label"] == "Real"


def test_unknown_rewritten_path_is_json_404(client):
    r = client.get("/api/index?__path=nope")
    assert r.status_code == 404 and r.get_json()["path"] == "/nope"
