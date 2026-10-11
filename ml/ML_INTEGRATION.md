# ML Module — Backend Integration Contract

## 1. Model Information
- Algorithm: Random Forest Classifier
- Number of trees: 200
- Random state: 42
- Model artifact: `models/predictive_maintenance_model_candidate.joblib`
- Decision threshold: 0.32
- Minimum recall target during threshold selection: 0.80

## 2. Prediction Input

The prediction function is available in `predict.py`:

`predict_failure(machine_data: dict) -> dict`

Required fields:

| Field | Expected type | Validation |
|---|---|---|
| `Type` | String | `L`, `M`, or `H` |
| `Air temperature [K]` | Number | Greater than 0 |
| `Process temperature [K]` | Number | Greater than 0 |
| `Rotational speed [rpm]` | Number | Greater than 0 |
| `Torque [Nm]` | Number | Zero or greater |
| `Tool wear [min]` | Number | Zero or greater |

All sensor readings must be finite numeric values. Booleans are not accepted as sensor readings.

## 3. Prediction Output

The function returns a JSON-friendly dictionary containing:
- `model_name`
- `failure_probability`
- `threshold`
- `predicted_failure`
- `status`

`predicted_failure` is determined by comparing the predicted failure probability with the saved threshold.

Possible status values:
- `FAILURE_ALERT`
- `NO_FAILURE_ALERT`

## 4. Integration Requirements
- Backend framework: Flask REST API.
- Load the saved model through the ML prediction module.
- Call `predict_failure()` from the prediction endpoint.
- Return the prediction response as JSON.
- Do not retrain the model for each API request.
- Do not create dummy production readings.
- Invalid input should return a clear client error.
- Keep the model artifact path consistent with the ML module.

## 5. Dataset and Limitations
- Dataset: UCI AI4I 2020 Predictive Maintenance Dataset.
- Training dataset contains 10,000 rows and 339 failure cases.
- The dataset is synthetic; real factory/IoT performance has not been established.
- The threshold was selected using out-of-fold predictions. Reported OOF metrics are not an independent final performance estimate.

## 6. Testing
Before integration, run:

`python -m pytest tests/test_predict_reliability.py -v`

The ML reliability tests should pass before the backend calls the prediction function.
