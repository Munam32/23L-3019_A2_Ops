import io

import pytest
from PIL import Image

from app import Prediction, create_app, db


class FakeClassifier:
    name = "fake-v0"
    threshold = 0.5

    def predict(self, data):
        return {"label": "AI-generated", "confidence": 0.9, "p_ai": 0.9, "threshold": 0.5, "model": self.name}


@pytest.fixture
def client():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite://", "MAX_CONTENT_LENGTH": 200_000},
                     classifier=FakeClassifier())
    with app.test_client() as c:
        yield c


def png_bytes(size=(32, 32)):
    buf = io.BytesIO()
    Image.new("RGB", size, "red").save(buf, "PNG")
    return buf.getvalue()


def upload(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["database"] == "ok"


def test_valid_upload_is_logged(client):
    r = upload(client, png_bytes(), "../../evil name!.png")
    assert r.status_code == 200
    assert b"AI-generated" in r.data
    with client.application.app_context():
        row = db.session.query(Prediction).one()
        assert row.filename == "evil_name.png"
        assert row.label == "AI-generated"
    assert b"evil_name.png" in client.get("/history").data


def test_wrong_extension_rejected(client):
    r = upload(client, b"GIF89a", "pic.gif")
    assert r.status_code == 400
    assert b"Only JPG and PNG" in r.data


def test_fake_image_rejected(client):
    r = upload(client, b"this is not an image", "pic.jpg")
    assert r.status_code == 400
    assert b"cannot be decoded" in r.data


def test_extension_mismatch_rejected(client):
    r = upload(client, png_bytes(), "pic.jpg")
    assert r.status_code == 400
    assert b"does not match" in r.data


def test_too_large_rejected(client):
    r = upload(client, png_bytes((600, 600)) + b"0" * 300_000, "big.png")
    assert r.status_code == 413
    assert b"File too large" in r.data


def test_failed_uploads_not_logged(client):
    upload(client, b"nope", "x.jpg")
    with client.application.app_context():
        assert db.session.query(Prediction).count() == 0
