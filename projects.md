---
title: "Projects"
author_profile: true
layout: collection
collection: projects
permalink: /projects/
entries_layout: grid  # Usa grid para una cuadrícula de proyectos
classes: wide
header:
   teaser: /assets/images/teaser.jpg
sidebar:
  title: "Projects"
  nav: "_data/projects-sidebar"
toc: true
toc_label: "Table of Contents"
toc_icon: "fas fa-list"
toc_sticky: true
toc_skin: "blue"
date: 2023-08-01
---

Below is a showcase of my Data Science and Machine Learning projects.

## My Projects

Below are some of my most recent and impactful projects in Data Science and Machine Learning.

### Featured Projects

1. **[Fraud Detection in Financial Transactions](/projects/fraud-detection)**  
   Developed a machine learning model that detects fraudulent transactions with over 95% accuracy.

2. **[Sentiment Analysis](/projects/sentiment-analysis)**  
   Created a customer segmentation model using clustering techniques to enhance targeted marketing.

<!-- 3. **[Time Series Forecasting for Sales Data](./projects/time-series-forecasting)**  
   Implemented a time series forecasting model to predict sales, reducing inventory costs by 25%. -->

# Preparation of the work environment

In this section, we will prepare our working environment by installing and loading the necessary libraries. Proper setup is essential for efficient data handling, visualization, and analysis in R. This step ensures that all the tools and libraries we need are readily available for our exploration and modeling tasks.

## Installing libraries

In this step, we install the necessary libraries for our analysis, including packages for data manipulation, visualization, text mining, and word clouds. Each package serves a unique purpose in the data science workflow, as detailed below:

-   **tidyverse**: For data manipulation and visualization.
-   **data.table**: Optimized for handling large datasets efficiently.
-   **ggplot2**: Essential for creating a wide variety of visualizations.
-   **dplyr**: Used for data wrangling and filtering.
-   **tm**: Text mining for natural language processing (NLP).
-   **wordcloud**: Generates word clouds for text data visualization.
-   **readr**: Efficient data reading.
-   **knitr**: Supports notebook creation and reporting.

```
# Install necessary libraries (only first time)
# install.packages(c("tidyverse", "data.table", "ggplot2", "dplyr", "tm", "wordcloud", "knitr", "tm"))
```