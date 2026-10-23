import importlib.util
import os
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS

# Locate the existing ML module without copying or retraining the model.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(PROJECT_ROOT, "ml")
PREDICT_PATH = os.path.join(ML_DIR, "predict.py")

if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

# Load the ML module explicitly from the project folder so the editor can resolve
# the dependency without requiring a package install or duplicate model files.
spec = importlib.util.spec_from_file_location("predict", PREDICT_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load ML predictor from '{PREDICT_PATH}'")

predict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_module)
predict_failure = predict_module.predict_failure
MODEL_NAME = predict_module.MODEL_NAME
THRESHOLD = predict_module.THRESHOLD


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
