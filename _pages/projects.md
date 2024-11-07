---
title: "Projects"
excerpt: "Explore a collection of my work across various domains in Data Science and Machine Learning, from Natural Language Processing to Real-Time Data Processing."
author_profile: false
layout: single
collection: projects
permalink: /projects/
entries_layout: grid  # Usa grid para una cuadrícula de proyectos
header:
   teaser: assets/images/projects.jpg
   overlay_image: assets/images/projects.jpg
   overlay_filter: "0.5"
   caption: "Photo credit: [**Unsplash**](https://unsplash.com)" 
sidebar:
  title: "Projects"
  nav: "projects-sidebar"
toc: true
toc_label: "Table of Contents"
toc_icon: "fas fa-list"
toc_sticky: true
toc_skin: "blue"


language-model-for-text-generation:
  - url: /posts/2016-07-20-text-generation
    image_path: assets/images/text-generator.jpg
    alt: "Text Generation"
    title: ""
    excerpt: "Trained a language model to generate human-like text based on input prompts, showcasing advanced NLP capabilities."
    btn_label: "Read More"
    btn_class: "btn--info"
sentiment-analysis:
  - url: /posts/2016-07-20-sentiment-analysis
    image_path: /assets/images/sentiments.jpg
    alt: "Sentiment Analysis"
    title: ""
    excerpt: "Built a sentiment analysis model to classify text as positive or negative, useful for customer feedback and social media analysis."
    btn_label: "Read More"
    btn_class: "btn--info"
language-model:
  - url: /posts/2016-07-20-language-model
    image_path: assets/images/language-model.jpg
    alt: "Language Model"
    title: ""
    excerpt: "Developed a custom language model for specific language tasks, improving predictive text capabilities."
    btn_label: "Read More"
    btn_class: "btn--info"
image-classification:
  - url: /posts/2016-07-20-image-classification
    image_path: assets/images/image-classification.jpg
    alt: "Image Classificarion"
    title: ""
    excerpt: "Developed a convolutional neural network (CNN) to classify images into predefined categories with high accuracy."
    btn_label: "Read More"
    btn_class: "btn--info"
image-recognition:
  - url: /posts/2016-07-20-image-recognition
    image_path: assets/images/image-recognition.jpg
    alt: "Image Recognition"
    title: ""
    excerpt: "Built an image recognition system to identify objects and landmarks in images, demonstrating skills in deep learning for computer vision."
    btn_label: "Read More"
    btn_class: "btn--info"
real-time-object-detection:
  - url: /posts/2016-07-20-object-detection
    image_path:  /assets/images/object-detection.jpg
    alt: "Real Time Object Detection"
    title: ""
    excerpt: "Developed a real-time object detection model using YOLO to identify and classify objects in video streams, optimized for high-speed processing."
    btn_label: "Read More"
    btn_class: "btn--info"
customer-recommendation-system:
  - url: /posts/2016-07-20-recomendation-system
    image_path:  /assets/images/recomendation-system.jpg
    alt: "Customer Recommendation System"
    title: ""
    excerpt: "Created a personalized recommendation system using collaborative filtering to enhance customer engagement for e-commerce."
    btn_label: "Read More"
    btn_class: "btn--info"
real-time-data-processing:
  - url: /posts/2016-07-20-real-time-data-processing
    image_path:  assets/images/real-time-data-processing.jpg
    alt: "Real-Time Data Processing"
    title: ""
    excerpt: "Designed a pipeline for processing data streams in real time, integrating Apache Kafka and Spark for scalable processing."
    btn_label: "Read More"
    btn_class: "btn--info"
fraud-detection:
  - url: /posts/2016-07-20-fraud-detection
    image_path:  /assets/images/fraud-detection.jpg
    alt: "Fraud Detection"
    title: ""
    excerpt: "Built a machine learning model to detect fraudulent transactions with high accuracy, leveraging ensemble methods like Random Forest and Gradient Boosting."
    btn_label: "Read More"
    btn_class: "btn--info"
use-patterns-analysis:
  - url: /posts/2016-07-20-use-patterns
    image_path:  assets/images/use-patterns.jpg
    alt: "Use Patterns Analysis"
    title: ""
    excerpt: "Analyzed user behavior data to identify patterns and trends, providing actionable insights for customer engagement."
    btn_label: "Read More"
    btn_class: "btn--info"
---

# Natural Language Processing (NLP)

## Language Model for Text Generation

{% include feature_row id="language-model-for-text-generation" type="left" %}
  
## Sentiment Analysis

{% include feature_row id="sentiment-analysis" type="left" %}

## Custom Language Model
  
{% include feature_row id="language-model" type="left" %}

# Computer Vision

## Image Classification

{% include feature_row id="image-classification" type="right" %}

## Image Recognition

{% include feature_row id="image-recognition" type="right" %}

## Object Detection System

{% include feature_row id="real-time-object-detection" type="right" %}

# Recommendation Systems

## Customer Recommendation System

{% include feature_row id="customer-recommendation-system" type="left" %}

# Real-Time Data Processing

## Real-Time Data Processing Pipeline

{% include feature_row id="real-time-data-processing" type="right" %}

# Data Patterns and Analytics

## Fraud Detection

{% include feature_row id="fraud-detection" type="left" %}

## Use Patterns Analysis

{% include feature_row id="use-patterns-analysis" type="left" %}