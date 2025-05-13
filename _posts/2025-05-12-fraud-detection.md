---
layout: single
title: "Fraud Detection with Machine Learning"
sidebar:
  title: "Projects"
  nav: "projects-sidebar"
excerpt: "Detecting fraudulent transactions using ensemble learning, FastAPI deployment, and automated testing."
header:
  teaser: "/assets/images/fraud-detection.jpg"  # Imagen de portada del proyecto
  overlay_image: "/assets/images/fraud-detection.jpg"  # Imagen de portada del proyecto
  overlay_filter: linear-gradient(rgba(255, 0, 0, 0.5), rgba(0, 255, 255, 0.5))
  overlay_filter: "0.5"
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
categories: [project]
  - project
tags:
  - Classification
  - Fraud Detection
  - FastAPI
  - Scikit-learn
  - XGBoost
  - Ensemble
date: 2025-05-12
toc: true
toc_sticky: true
toc_label: "Contents"
related: false
share: false
---

## 📌 Project Overview

This project implements a complete machine learning pipeline to detect fraudulent transactions using the [IEEE-CIS dataset](https://www.kaggle.com/c/ieee-fraud-detection). It includes:

- Deep EDA in **R** and **Python**
- Robust preprocessing and feature engineering
- Ensemble modeling (XGBoost, LightGBM, CatBoost)
- A FastAPI deployment for real-time predictions
- Unit testing with `pytest` for pipeline robustness

🚀 Check the full repository: [GitHub](https://github.com/alexmatiasas/Fraud-Detection-with-ML)

---

## 🧠 Dataset and Preprocessing

The dataset contains transaction and identity features, most of them anonymized.

Key steps:

- Merged identity and transaction data
- Imputed missing values using statistical strategies
- Encoded categorical variables using `LabelEncoder`
- Scaled numerical features using `StandardScaler`
- Saved transformers and models using `joblib` for reuse in deployment

---

## 🤖 Model Training and Evaluation

We trained and compared multiple models:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- **Stacking Ensemble** with Logistic Regression as meta-learner

All models were evaluated using:

- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Confusion matrices
- ROC curves

📊 Results are available in the [reports section](https://github.com/alexmatiasas/Fraud-Detection-with-ML/tree/main/reports/figures).

---

## 🧪 Testing

We implemented tests to ensure pipeline integrity:

- ✅ Preprocessing pipeline does not crash with valid data
- ✅ API responds with expected output structure
- ✅ Unit tests managed with `pytest`

---

## 🖥️ Deployment

The final model was deployed using **FastAPI**. Key features:

- `/predict` endpoint returns prediction and fraud probability
- Interactive Swagger UI available at `/docs`
- Can be run locally via Uvicorn or deployed in a Docker container

```bash
uvicorn src.main:app --reload
```

Or with Docker

```bash
docker pull alexmatiasastorga/fraud-api:latest
docker run -d -p 8000:8000 alexmatiasastorga/fraud-api
```

## 📌 Conclusion

This project demonstrates a real-world machine learning workflow from raw data to deployment. Future improvements may include:

- DAG automation with Apache Airflow
- Cloud deployment (Render or AWS)
- Monitoring with MLFlow or Prometheus

## 🔗 Links

- 🔍 [Project Repository](https://github.com/alexmatiasas/Fraud-Detection-with-ML)
- 🐳 [Docker Image](https://hub.docker.com/r/alexmatiasastorga/fraud-api)