# ⚙️ Predictive Maintenance AI

An end-to-end machine learning system for predicting industrial machine
failures using sensor and operating-condition data.

The project uses the AI4I 2020 Predictive Maintenance Dataset and compares
multiple classification algorithms before deploying a Random Forest model
through a Streamlit web application.

---

## 🚀 Project Overview

Unexpected machine failures can cause production downtime and maintenance
costs.

This project explores whether machine operating parameters can be used to
predict potential failures before they occur.

The system takes the following parameters as input:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

and produces:

- Failure probability
- Failure/normal prediction
- Model information
- Feature importance

---

## 📊 Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset.

Dataset characteristics:

- 10,000 observations
- 14 original columns
- Binary machine-failure target
- Highly imbalanced target distribution
- 339 machine failures
- 9,661 non-failures

The following identifiers were excluded:

- UDI
- Product ID

The failure-mode indicators were also excluded from model training:

- TWF
- HDF
- PWF
- OSF
- RNF

These variables were excluded because they are directly associated with
machine failure and could introduce target leakage.

---

## 🧠 Machine Learning Pipeline

The project follows this pipeline:

Dataset
↓
Data exploration
↓
Feature selection
↓
Train/Test Split
↓
Preprocessing
↓
Model Training
↓
Model Comparison
↓
Random Forest Selection
↓
Threshold Analysis
↓
Streamlit Deployment

---

## 🤖 Models Compared

Three classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Because machine failure is an imbalanced classification problem,
accuracy alone was not used to select the final model.

Precision, Recall and F1-score were also evaluated.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.70% | 14.36% | 82.35% | 24.45% |
| Decision Tree | 95.30% | 40.58% | 82.35% | 54.37% |
| Random Forest | 97.95% | 71.43% | 66.18% | 68.70% |

Random Forest achieved the highest F1-score among the evaluated models.

---

## 🎯 Threshold Analysis

The default binary classification threshold was initially 0.50.

Threshold analysis was performed to study the precision-recall trade-off.

At a threshold of 0.45:

- Precision: 66.22%
- Recall: 72.06%
- F1-score: 69.01%

Compared with the 0.50 threshold:

- Recall increased from 66.18% to 72.06%
- F1-score increased from 68.70% to 69.01%

The application therefore uses a classification threshold of 0.45.

---

## 🔍 Feature Importance

The Random Forest model identified the following features as the most
important predictive variables:

| Feature | Importance |
|---|---:|
| Torque | 30.59% |
| Rotational Speed | 29.74% |
| Tool Wear | 20.94% |
| Air Temperature | 10.07% |
| Process Temperature | 6.88% |

Machine Type contributed relatively little to the model.

Feature importance represents predictive contribution and should not be
interpreted as causal influence.

---

## 📈 Exploratory Data Analysis

The project includes analysis of:

- Machine failure distribution
- Torque vs machine failure
- Rotational speed vs machine failure
- Tool wear vs machine failure
- Feature correlation

Important observations included:

- The dataset is highly imbalanced.
- Failed machines showed higher torque values.
- Failed machines showed higher tool wear.
- Failed machines had a different rotational-speed distribution.
- Torque and rotational speed showed a strong negative correlation.

---

## 🖥️ Streamlit Application

The trained model is deployed through a Streamlit application.

Users can enter machine operating parameters and receive:

- Failure probability
- Machine failure prediction
- Input summary
- Model performance information
- Feature importance

Run the application with:

```bash
streamlit run app.py