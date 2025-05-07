# 🔍 Fake News Detection (NLP)

## 📊 Overview

Misinformation is one of the most pressing global challenges we face today. While incorrect information can be harmless in some contexts, it can have devastating consequences in others.

In this project, we are provided with approximately 45,000 news articles, each containing a title, text, subject, publication date, and label. The objective is to classify the news as either fake or real using these features.

## 📚 Dataset

Data: [Download Link](https://drive.google.com/file/d/1CZzfZDvE5E7HaHjk9yeyZDKil4_jkass/view?usp=drive_link)

Since the data is sourced from various news outlets, it contains not only unwanted text, such as injected JavaScript code for website display, but also source-specific patterns, like news articles beginning with "Reuters (City Name)" or those referencing video or image sources. Therefore, data cleaning is a critical step to ensure the text is semantically meaningful and to prevent data leakage from source-specific patterns. Models trained on well-cleaned data are also more likely to generalize effectively to external datasets.

## 🎯 Project Goal

Our primary goal is on building and comparing different models to predict whether a news article is fake. Some models incorporate heuristic features, such as text length, punctuation counts, and basic sentiment analysis using the Vader sentiment score. Others leverage features derived from text-to-vector methods like TF-IDF Vectorization and Word2Vec embeddings. Additionally, we will explore more advanced Transformer models. We will evaluate and discuss the trade-offs of these approaches and recommend the most suitable model for this project.

## 📑 Analysis Notebooks

- [Cleaning 📚](https://github.com/MeiChieh/fake-news-detection/blob/main/1_cleaning.ipynb)
- [EDA 📚](https://github.com/MeiChieh/fake-news-detection/blob/main/2_eda.ipynb)
- [Modeling 📚](https://github.com/MeiChieh/fake-news-detection/blob/main/3_modeling.ipynb)

## 🔄 Analysis Structure

### 🧹 Cleaning

- Remove empty and short texts.
- Format date and subjects columns.
- Remove or replace source specific and spoiler texts.
- Split data into train, val, test.

### 📊 EDA and Feature Engineering

1. Label and Feature Distribution

   - Target label
   - News subject
   - News sequence length
   - Published date
   - Word Frequency with wordcloud

2. Sentiment Analysis

   - Uppercase tokens and sentimental punctuations
   - Sentiment polarity analysis with Vader

3. Correlation among features and labels

4. Text vectorization and embedding methods
   - TF-IDF Vectorizer
   - Word2Vec

### 🤖 Modeling

1. Model selection and reasoning

2. Modeling

   - preparation: class weight calculation, find initial lr
   - training

3. Compare Transformer Models with Baseline Models

   - Heuristic Features Model
   - TF-IDF Features Model
   - Validation Metrics Comparison
   - Test Prediction Comparison

4. Error analysis with Lime

## 📂 Project Structure

```
├── 1_cleaning.ipynb      # Data cleaning and preprocessing
├── 2_eda.ipynb          # Exploratory data analysis
├── 3_modeling.ipynb     # Model development and evaluation
├── helper/              # Helper functions and utilities
│   ├── __init__.py
│   ├── project_helper_functions.py  # General project utilities
│   ├── stopwords.py                # Custom stopwords list
│   ├── custom_transformers.py      # Custom transformer implementations
│   ├── helper_function.py          # Basic helper functions
│   └── nlp_helper_functions.py     # NLP-specific utilities
└── README.md            # Project documentation
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Transformers
- Vader Sentiment Analysis
- LIME (Local Interpretable Model-agnostic Explanations)

## 🚀 Setup and Installation

1. Clone the repository
2. Install dependencies:

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
