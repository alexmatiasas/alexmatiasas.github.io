---
title: "Sentiment Analysis on IMDb Reviews"
excerpt: "Applied NLP techniques to classify IMDb reviews by sentiment using R and Python. Includes EDA, tokenization, visualization, and plans for classification models and deployment."
layout: single
classes: wide
author_profile: true
read_time: true
related: true
share: true
categories: [project]
  - Natural Language Processing (NLP)
tags:
  - Sentiment Analysis
  - Natural Language Processing (NLP)
  - Text Processing
  - Classification
  - Scikit-Learn
  - Python
  - R
---

## 🎯 Project Overview

This project analyzes a dataset of **[50,000 IMDb movie reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)**, each labeled as _positive_ or _negative_. The goal is to classify sentiment through a two-phase workflow:

1. **Exploratory Data Analysis and preprocessing in R**  
2. **Sentiment classification modeling and deployment in Python**

The first phase leverages R’s rich ecosystem for text manipulation and data visualization. The second phase (in progress) will involve training ML models and deploying a classifier using Python.

---

## 🛠️ Tools & Libraries

- **R**: `tidyverse`, `tidytext`, `ggplot2`, `udpipe`, `SnowballC`, `textstem`, `DT`
- **Python** (planned): `scikit-learn`, `nltk`, `pandas`, `matplotlib`, `PyTorch`
- **Techniques**: Tokenization, POS tagging, stemming, lemmatization, n-grams, polarity
- **Format**: R Markdown (`.Rmd`) → HTML (via RPubs)

---

## 📊 Key Explorations

- Sentiment class distribution and review length analysis
- HTML cleaning, stopword removal, punctuation & casing normalization
- **POS tagging** using `udpipe`
- **Stemming** (`SnowballC`) and **lemmatization** (`textstem`)
- **N-gram analysis** for phrase structure insight

---

## 📈 Visualizations Preview

Some of the visualizations in the RPubs notebook include:

- Word frequency lollipop charts
- Sentiment-based word clouds
- N-gram distribution (bigrams & trigrams)
- Polarity sentiment barplots

➡️ [Explore the full interactive report here](https://rpubs.com/tu_usuario/tu_publicacion)

---

## 📘 Full Exploratory Report

🔗 [View the full EDA on RPubs](https://rpubs.com/tu_usuario_o_url_del_EDA)  
_(Hosted via RStudio's RPubs; includes interactive visuals and data breakdown)_

---

## 🔮 Next Steps

- Export cleaned data to `.csv` for model training
- Build classifiers using:
  - Logistic Regression & Naive Bayes (baseline)
  - Pipeline-based ML models (`scikit-learn`)
  - Deep learning model using `PyTorch` (planned)
- Evaluation: **Confusion Matrix**, **F1 Score**, **ROC-AUC**
- Optionally deploy via **Streamlit** or **Apache Spark**

---

## 🧾 Deliverables

- [`sentiment-analysis.Rmd`](https://rpubs.com/tu_usuario/tu_publicacion) — Core analysis (published on RPubs)
- `IMDB-cleaned.csv` — Cleaned dataset (not public due to size, available on request)
- [ ] `model_sentiment.py` — Machine learning model (coming soon)
- [ ] Streamlit dashboard or deployment interface (planned)
- [GitHub repository (in process)](https://github.com/alexmatiasas/Sentiment-Analysis)

---

## 📌 Outcome

> Completed a robust EDA and text processing pipeline in R.  
> Laying the foundation for cross-platform sentiment classification with Python.

---

## 🧠 What I Learned

- R is powerful for quick and elegant EDA and text visualization.
- Handling natural language data requires both linguistic and statistical intuition.
- Preprocessing choices (e.g., stemming vs lemmatization) can deeply affect downstream model performance.

🔗 [View this project on GitHub](https://github.com/alexmatiasas/Sentiment-Analysis)

---

_Last updated: 2025-04-20_