# Fake News Detection (NLP)

## Overview

Misinformation is one of the most pressing global challenges we face today. While incorrect information can be harmless in some contexts, it can have devastating consequences in others.

In this project, we are provided with approximately 45,000 news articles, each containing a title, text, subject, publication date, and label. The objective is to classify the news as either fake or real. 

We will build and evaluate different models, discussing their trade-offs in terms of computational resources, training time, and performance to help inform model selection decisions.

## Dataset

Data: [Download Link](https://drive.google.com/file/d/1CZzfZDvE5E7HaHjk9yeyZDKil4_jkass/view?usp=drive_link)

Since the data is sourced from various news outlets, it contains not only unwanted text, such as injected JavaScript code for website display, but also source-specific patterns, like news articles beginning with "Reuters (City Name)" or those referencing video or image sources. 

Therefore, data cleaning is a critical step to ensure the text is semantically meaningful and to prevent data leakage from source-specific patterns. Models trained on well-cleaned data are also more likely to generalize effectively to external datasets.

## Results

Our analysis identified three top-performing models, each demonstrating distinct trade-offs between computational complexity and predictive performance:

- Logistic Regression + TF-IDF Features
- Cased DistilBERT Model
- Cased Longformer Model

| Model | Accuracy | Recall | Training Time |
|-------|----------|--------|---------------|
| Logistic Regression + TF-IDF | 0.979 | 0.972 | 45s |
| Cased DistilBERT | 0.998 | 0.997 | 9m 39s |
| Cased Longformer | 0.992 | 0.997 | 79m |


1. **TF-IDF + Logistic Regression (97.9% accuracy)**
   - Fastest training (45s)
   - Minimal computational resources
   - Only 2-3% lower accuracy than transformers
   - Best for: Organizations with limited compute resources or need for quick deployment

2. **DistilBERT (99.8% accuracy)**
   - Moderate training time (9.5 mins)
   - Best accuracy-to-time ratio
   - Handles 97% of news articles effectively
   - Best for: Organizations requiring high accuracy with reasonable computational costs

3. **Longformer (99.2% accuracy)**
   - Longest training time (79 mins)
   - Similar performance to DistilBERT
   - Can process longer sequences
   - Best for: Specialized cases requiring analysis of very long articles

For most business applications, we recommend:
- **Cost-sensitive**: TF-IDF model - provides excellent performance with minimal resources
- **Balanced approach**: DistilBERT - optimal balance of accuracy and computational cost
- **Specialized needs**: Longformer - only if handling exceptionally long articles is crucial



## Analysis Notebooks

- [Cleaning 📚](https://github.com/MeiChieh/fake-news-detection/blob/main/1_cleaning.ipynb)
- [EDA 📚](https://github.com/MeiChieh/fake-news-detection/blob/main/2_eda.ipynb)
- [Modeling 📚](https://github.com/MeiChieh/fake-news-detection/blob/main/3_modeling.ipynb)

## Analysis Structure

### Data Cleaning

- Remove empty and short texts.
- Format date and subjects columns.
- Remove or replace source specific and spoiler texts.
- Split data into train, val, test.

### EDA and Feature Engineering

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

### Modeling

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

## Project Structure

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

## Tech Stack

- Python
- Pytorch
- Pandas
- NumPy
- Scikit-learn
- HuggingFace
- Transformers
- Vader Sentiment Analysis
- LIME (Local Interpretable Model-agnostic Explanations)

## 🚀 Setup and Installation

1. Clone the repository
2. Install dependencies:

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
