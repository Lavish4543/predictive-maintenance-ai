import os
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS

# Locate the existing ML module without copying or retraining the model.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(PROJECT_ROOT, "ml")

 
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from predict import predict_failure, MODEL_NAME, THRESHOLD  # noqa: E402


app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "predictive-maintenance-backend",
        "model_name": MODEL_NAME,
    }), 200


@app.post("/predict")
def predict():
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json."
        }), 415

    machine_data = request.get_json(silent=True)

    if not isinstance(machine_data, dict):
        return jsonify({
            "error": "Request body must be a valid JSON object."
        }), 400

    try:
        result = predict_failure(machine_data)
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
