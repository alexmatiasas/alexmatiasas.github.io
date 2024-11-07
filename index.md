---
layout: splash
title: "Welcome to My Data Science Portfolio!"
excerpt: "Explore my projects, blog posts, and case studies."
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
      url: "CV/Manuel Alejandro Matías Astorga CV ENG.pdf"
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
    excerpt: "Explore a collection of my work across various domains in Data Science and Machine Learning, from Natural Language Processing to Real-Time Data Processing."
    btn_label: "Go to Project Section"
    btn_class: "btn--primary"
  - url: /Blog
    image_path:  assets/images/blog.jpg
    alt: "Blog"
    title: "Blog"
    excerpt: "Insights and tutorials on Data Science, Machine Learning, and more."
    btn_label: "Go to my Blog "
    btn_class: "btn--primary"
  - url: /about
    image_path:  assets/images/About me.jpg
    alt: "About Me"
    title: "About Me"
    excerpt: "Learn more about my background in Data Science and Physics."
    btn_label: "About Me"
    btn_class: "btn--primary"

fraud-detection:
  - url: /posts/2016-07-20-fraud-detection
    image_path:  /assets/images/fraud-detection.jpg
    alt: "Fraud Detection"
    title: "Fraud Detection"
    excerpt: "Built a machine learning model to detect fraudulent transactions with high accuracy, leveraging ensemble methods like Random Forest and Gradient Boosting."
    btn_label: "Read More"
    btn_class: "btn--info"
real-time-object-detection:
  - url: /posts/2016-07-20-object-detection
    image_path:  /assets/images/object-detection.jpg
    alt: "Real Time Object Detection"
    title: "Real Time Object Detection"
    excerpt: "Developed a real-time object detection model using YOLO to identify and classify objects in video streams, optimized for high-speed processing."
    btn_label: "Read More"
    btn_class: "btn--info"
customer-recommendation-system:
  - url: /posts/2016-07-20-recomendation-system
    image_path:  /assets/images/recomendation-system.jpg
    alt: "Customer Recommendation System"
    title: "Customer Recommendation System"
    excerpt: "Created a personalized recommendation system using collaborative filtering to enhance customer engagement for e-commerce."
    btn_label: "Read More"
    btn_class: "btn--info"
language-model-for-text-generation:
  - url: /posts/2016-07-20-text-generation
    image_path: assets/images/text-generator.jpg
    alt: "Text Generation"
    title: "Text Generation"
    excerpt: "Trained a language model to generate human-like text based on input prompts, showcasing advanced NLP capabilities."
    btn_label: "Read More"
    btn_class: "btn--info"
  # - image_path: /assets/images/unsplash-gallery-image-3-th.jpg
  #   title: "Placeholder 3"
  #   excerpt: "This is some sample content that goes here with **Markdown** formatting."
---

## Hi, I'm Alejandro!

{: .text-justify}
I am a **Data Scientist** and **Machine Learning Engineer** with a background in **Physics** (PhD) and a passion for leveraging data to solve complex real-world problems. My expertise lies at the intersection of machine learning, data analysis, and scientific research, and I thrive on translating raw data into actionable insights.

{% include feature_row id="home_feature_row"%}

# <i class="fas fa-project-diagram"></i> Featured projects

Here are some of my most impactful projects. For a full list, visit my [Projects](./projects) page.

<!-- {% include gallery caption="This is a sample gallery with **Markdown support**." %} -->
{% include feature_row id="fraud-detection" type="left" %}
{% include feature_row id="real-time-object-detection" type="left" %}
{% include feature_row id="customer-recommendation-system" type="left" %}
{% include feature_row id="language-model-for-text-generation" type="left" %}