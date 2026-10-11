# Predictive Maintenance AI — ML Evaluation Report

## 1. Project Overview

The project uses machine sensor readings to predict machine failure risk using a supervised machine learning classifier.

**Dataset:** AI4I 2020 Predictive Maintenance Dataset  
**Dataset size:** 10,000 records  
**Target:** `Machine failure`  
**Candidate model:** Random Forest Classifier  
**Primary selection objective:** Achieve at least 80% recall on threshold-selection data, then maximize precision among eligible thresholds.

The dataset is synthetic. The current results demonstrate model development and software testing, not proven reliability on physical industrial machines.

## 2. Data Preparation

The following columns are excluded from model training:

- `UDI`
- `Product ID`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

The failure-mode columns are excluded to prevent target leakage. `Type` is one-hot encoded into `Type_H`, `Type_L`, and `Type_M`.

The model uses five sensor features:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

The dataset contains 339 failure records and 9,661 normal records, making the target class imbalanced.

## 3. Candidate Model Comparison

Five models were compared using stratified cross-validation and out-of-fold predictions.

| Model | Precision | Recall | F1-score | Average Precision |
|---|---:|---:|---:|---:|
| Random Forest | 0.5302 | 0.8024 | 0.6385 | 0.7519 |
| Gradient Boosting | 0.4218 | 0.8112 | 0.5550 | 0.6657 |
| Extra Trees | 0.3835 | 0.8112 | 0.5208 | 0.7166 |
| Logistic Regression | 0.1391 | 0.8024 | 0.2370 | 0.4236 |
| Decision Tree | 0.1469 | 0.7699 | 0.2467 | 0.5581 |

These are development-stage comparison results. Threshold selection and reporting used the same out-of-fold prediction set, so the metrics may be optimistic and should not be presented as an independent final performance estimate.

Random Forest was selected as the leading candidate because it provided the strongest precision and F1-score among the compared models while meeting the recall target in this comparison.

## 4. Nested Cross-Validation

A separate nested cross-validation experiment was conducted for Random Forest.

- Outer folds: 5
- Inner folds: 4
- Threshold selection: inner-fold out-of-fold predictions
- Evaluation: held-out outer fold
- Recall target during threshold selection: 80%

### Mean outer-fold results

| Metric | Mean |
|---|---:|
| Precision | 0.5044 |
| Recall | 0.8172 |
| F1-score | 0.6223 |
| Average Precision | 0.7532 |
| ROC-AUC | 0.9678 |

The threshold selected within each outer fold was 0.26, 0.30, 0.33, 0.28, and 0.28.

The recall target was met during threshold selection in all five outer folds. However, recall on the held-out outer folds varied, and not every fold achieved 80% recall.

This shows that meeting a validation target does not guarantee the same recall on unseen data.

## 5. Candidate Model Configuration

The current candidate uses:

- Algorithm: Random Forest Classifier
- Number of trees: 200
- Class weighting: `balanced`
- Random state: 42
- Selected development threshold: 0.32
- Training records: 10,000

The candidate bundle is saved as:

`ml/models/predictive_maintenance_model_candidate.joblib`

The threshold of 0.32 was selected from out-of-fold predictions to meet the recall target and maximize precision among eligible thresholds.

Because threshold selection and the reported comparison use related development data, the reported metrics are not an independent final estimate of performance.

## 6. Software Validation

Automated tests have been run against the prediction module.

The tests cover:

- Valid input and output structure
- Probability range
- Threshold and prediction consistency
- Status and predicted-class consistency
- Missing required fields
- Invalid machine type
- Non-numeric sensor readings
- NaN and infinite values
- Invalid physical input values
- Repeatability for identical inputs
- Non-dictionary requests
- Boolean sensor readings

The test suite passed in the current development environment.

Passing software tests verifies expected application behavior. It does not establish industrial predictive reliability.

## 7. Limitations

1. AI4I 2020 is a synthetic dataset.
2. The model has not been validated on real factory sensor readings and verified maintenance outcomes.
3. The 80% recall target is a project objective, not a guaranteed operational performance level.
4. False alarms can cause unnecessary inspections and maintenance costs.
5. Missed failures can lead to downtime or equipment damage.
6. Probability calibration has not yet been established.
7. The model uses only the available five sensor features and machine type; it does not currently use vibration or pressure readings.
8. IoT integration is future scope.

## 8. Next Steps

1. Freeze the prediction input/output interface for backend integration.
2. Verify that the Flask API loads the intended candidate model.
3. Test the API with valid and invalid requests.
4. Evaluate probability calibration and document alert costs.
5. Obtain real machine data for independent validation when available.
6. Monitor performance during any future shadow deployment before using predictions for operational decisions.

## 9. Conclusion

Random Forest is the leading development candidate among the models evaluated. It provides a promising precision–recall balance on the AI4I dataset and has passed the current prediction software tests.

It should be described as a development-stage predictive maintenance model, not as a proven industrial system, until independent validation on real machine data is completed.