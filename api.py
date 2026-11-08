import hmac
import os
import time

from flask import Flask, jsonify, request

from validation import UNDECODABLE, read_upload


def create_api(classifier=None):
    key = os.getenv("API_KEY")
    if not key:
        raise RuntimeError("API_KEY is not set; refusing to start an unauthenticated API.")

    api = Flask(__name__)
    api.config["MAX_CONTENT_LENGTH"] = int(float(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024)
    if classifier is None:
        from inference import Classifier
        classifier = Classifier()

    @api.get("/health")
    def health():
        return jsonify(status="ok", model=classifier.name, threshold=classifier.threshold)

    @api.post("/predict")
    def predict():
        supplied = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied.encode(), key.encode()):
            return jsonify(error="invalid or missing API key"), 401
        try:
            name, data = read_upload(request.files.get("image"))
            started = time.perf_counter()
            try:
                result = classifier.predict(data)
            except Exception:
                raise ValueError(UNDECODABLE) from None
        except ValueError as err:
            return jsonify(error=str(err)), 400
        result["filename"] = name
        result["latency_ms"] = int((time.perf_counter() - started) * 1000)
        return jsonify(result)

    @api.errorhandler(413)
    def too_large(_):
        return jsonify(error="file too large"), 413

    return api
