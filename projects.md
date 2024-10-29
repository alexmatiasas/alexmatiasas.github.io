---
layout: collection
title: "Projects"
author_profile: true
collection: projects
permalink: /projects/
entries_layout: grid  # Usa grid para una cuadrícula de proyectos
classes: wide
sidebar:
  title: "Projects"
  nav: "projects-sidebar"
toc: true
toc_label: "Table of Contents"
toc_icon: "fas fa-list"
toc_sticky: true
toc_skin: "blue"
---

Below is a showcase of my Data Science and Machine Learning projects.

## My Projects

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