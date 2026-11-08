import base64
import io
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from sqlalchemy import text

from validation import UNDECODABLE, read_upload

load_dotenv()

db = SQLAlchemy()


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False,
                           default=lambda: datetime.now(timezone.utc))
    filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    label = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    p_ai = db.Column(db.Float, nullable=False)
    threshold = db.Column(db.Float, nullable=False)
    model = db.Column(db.String(120), nullable=False)
    latency_ms = db.Column(db.Integer, nullable=False)


def thumbnail_uri(data):
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((480, 480))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def create_app(config=None, classifier=None):
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
    app.config["MAX_CONTENT_LENGTH"] = int(float(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024)
    app.config.update(config or {})
    db.init_app(app)

    if classifier is None:
        from inference import Classifier
        classifier = Classifier()
    with app.app_context():
        db.create_all()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        try:
            name, data = read_upload(request.files.get("image"))
            started = time.perf_counter()
            try:
                result = classifier.predict(data)
            except Exception:
                raise ValueError(UNDECODABLE) from None
            latency = int((time.perf_counter() - started) * 1000)
        except ValueError as err:
            return render_template("index.html", error=str(err)), 400
        row = Prediction(filename=name, file_size=len(data), label=result["label"],
                         confidence=result["confidence"], p_ai=result["p_ai"],
                         threshold=result["threshold"], model=result["model"], latency_ms=latency)
        db.session.add(row)
        db.session.commit()
        return render_template("result.html", r=result, row=row, preview=thumbnail_uri(data))

    @app.get("/history")
    def history():
        rows = Prediction.query.order_by(Prediction.id.desc()).limit(200).all()
        return render_template("history.html", rows=rows)

    @app.get("/health")
    def health():
        try:
            db.session.execute(text("SELECT 1"))
            database = "ok"
        except Exception:
            database = "down"
        body = {"status": "ok" if database == "ok" else "degraded", "model": classifier.name,
                "threshold": classifier.threshold, "database": database}
        return jsonify(body), 200 if database == "ok" else 503

    @app.errorhandler(413)
    def too_large(_):
        limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return render_template("error.html", code=413, title="File too large",
                               message=f"Images must be {limit} MB or smaller."), 413

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, title="Page not found",
                               message="That page does not exist."), 404

    @app.errorhandler(500)
    def server_error(_):
        return render_template("error.html", code=500, title="Something went wrong",
                               message="The server hit an unexpected problem. Please try again."), 500

    return app


if __name__ == "__main__":
    create_app().run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)
