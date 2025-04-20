---
layout: splash
title: "Welcome to My Data Science Portfolio!"
excerpt: "Explore real-world projects and insights in Data Science, Machine Learning, and AI."
permalink: /
author_profile: true
# sidebar:
#   nav: "index"
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: /assets/images/teaser.jpg
  actions:
    - label: "Download my CV"
      url: "CV/CV_Data_science.pdf"
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
classes:
  - landing
  - wide
# gallery:
#   - url: /posts/2016-07-20-fraud-detection
#     image_path: /assets/images/fraud-detection.jpg
#     alt: "fraud-detection"
#     title: "Fraud Detection in Financial Transactions"
#   - url: /posts/2016-07-20-sentiment-analysis
#     image_path: /assets/images/teaser.jpg
#     alt: "sentiment-analysis"
#     title: "Sentiment Analysis"
#   - url: /assets/images/unsplash-gallery-image-3.jpg
#     image_path: /assets/images/unsplash-gallery-image-3-th.jpg
#     alt: "placeholder image 3"
#     title: "Image 3 title caption"

home_feature_row:
  - url: /projects
    image_path:  assets/images/projects.jpg
    alt: "Projects"
    title: "Projects"
    excerpt: "Explore a growing collection of real-world applications in Data Science, from Sentiment Analysis to Fraud Detection."
    btn_label: "View Projects"
    btn_class: "btn--primary"
  - url: /blog
    image_path:  assets/images/blog.jpg
    alt: "Blog"
    title: "Blog"
    excerpt: "Technical breakdowns, insights, and tutorials from my journey in ML and Data Science."
    btn_label: "Read Blog "
    btn_class: "btn--primary"
  - url: /about
    image_path:  assets/images/About me.jpg
    alt: "About Me"
    title: "About Me"
    excerpt: "Get to know more about my academic background and transition into Data Science."
    btn_label: "About Me"
    btn_class: "btn--primary"

fraud-detection:
  - url: /posts/2016-07-20-fraud-detection
    image_path: /assets/images/fraud-detection.jpg
    alt: "Fraud Detection"
    title: "Fraud Detection"
    excerpt: "Placeholder for upcoming project: ML model to detect fraudulent transactions using ensemble methods."
    btn_label: "Coming Soon"
    btn_class: "btn--light"
real-time-object-detection:
  - url: /posts/2016-07-20-object-detection
    image_path: /assets/images/object-detection.jpg
    alt: "Real Time Object Detection"
    title: "Real-Time Object Detection"
    excerpt: "Placeholder for YOLO-based real-time object detection pipeline on video streams."
    btn_label: "Coming Soon"
    btn_class: "btn--light"
customer-recommendation-system:
  - url: /posts/2016-07-20-recomendation-system
    image_path: /assets/images/recomendation-system.jpg
    alt: "Recommendation System"
    title: "Recommendation System"
    excerpt: "Placeholder for collaborative filtering recommendation engine."
    btn_label: "Coming Soon"
    btn_class: "btn--light"
language-model-for-text-generation:
  - url: /posts/2016-07-20-text-generation
    image_path: assets/images/text-generator.jpg
    alt: "Text Generation"
    title: "Text Generation"
    excerpt: "Placeholder for NLP language model generating human-like text."
    btn_label: "Coming Soon"
    btn_class: "btn--light"
  # - image_path: /assets/images/unsplash-gallery-image-3-th.jpg
  #   title: "Placeholder 3"
  #   excerpt: "This is some sample content that goes here with **Markdown** formatting."
---

## Hi, I'm Alejandro!

{: .text-justify}
I'm a **Data Scientist** and **Machine Learning Engineer** with a PhD in **Physics**, specializing in using data and code to solve complex problems. My expertise blends scientific research with modern machine learning practices, and I'm currently transitioning my academic rigor into scalable, data-driven solutions.

<p align="center">
  <a class="btn btn--primary" href="mailto:alejandromatiasastorga@gmail.com">Contact Me</a>
  <a class="btn btn--inverse" href="https://www.linkedin.com/in/alexmatiasastorga/" target="_blank">Connect on LinkedIn</a>
</p>

{% include feature_row id="home_feature_row"%}

# <i class="fas fa-project-diagram"></i> Featured projects

These are placeholders for my in-progress portfolio. Visit the [Projects](./projects) page for updates as they are completed.

{% include feature_row id="fraud-detection" type="left" %}
{% include feature_row id="real-time-object-detection" type="left" %}
{% include feature_row id="customer-recommendation-system" type="left" %}
{% include feature_row id="language-model-for-text-generation" type="left" %}

<div class="notice--info">
  <strong>Note:</strong> These projects are currently under active development. I’m building this portfolio to reflect real, practical experience in Data Science and Machine Learning. Stay tuned for updates!
</div>