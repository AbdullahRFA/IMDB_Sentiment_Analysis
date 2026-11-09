# IMDB Movie Review Sentiment Analysis

## Overview

This project focuses on **sentiment analysis of IMDB movie reviews** using Natural Language Processing (NLP) and Machine Learning techniques. The objective is to classify movie reviews as either positive or negative and compare the performance of three different text representation approaches.

## Objectives

- Perform sentiment classification on IMDB movie reviews.
- Apply text preprocessing and feature extraction techniques.
- Implement and compare TF-IDF, Word2Vec, and BERT representations.
- Evaluate models using standard classification metrics.
- Analyze prediction performance, training time, and computational requirements.

## Dataset

- **Dataset:** Stanford IMDB Movie Reviews
- **Source:** [Hugging Face – Stanford NLP IMDB](https://huggingface.co/datasets/stanfordnlp/imdb)
- **Total Reviews:** 50,000
- **Training Set:** 22,500 reviews
- **Validation Set:** 2,500 reviews
- **Test Set:** 25,000 reviews
- **Classes:** Positive (1) and Negative (0)

The original training set was divided into training and validation subsets, while the original test set was reserved for final evaluation.

## Methodology

The project follows these main steps:

1. **Data Loading:** Load the IMDB dataset and create training, validation, and test sets.
2. **Text Preprocessing:** Clean HTML tags, normalize text, remove unnecessary characters, and tokenize reviews where required.
3. **Feature Extraction:** Transform text into numerical representations using three approaches:
   - **TF-IDF:** Extract unigram and bigram features using `TfidfVectorizer`.
   - **Word2Vec:** Generate Skip-gram word embeddings and represent each review using average word vectors.
   - **BERT:** Extract contextual embeddings using `bert-base-uncased` with mean pooling.
4. **Classification:** Train Logistic Regression classifiers using the extracted features. BERT is used as a frozen feature extractor rather than fine-tuned.
5. **Evaluation:** Compare the models using accuracy, precision, recall, F1-score, training time, and prediction time.

## Technologies Used

- Python
- Google Colab
- Pandas and NumPy
- Scikit-learn
- Gensim
- Hugging Face Transformers
- PyTorch
- Matplotlib

## Final Test Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | **89.89%** | 89.52% | **90.36%** | **89.94%** |
| Word2Vec + Logistic Regression | 85.68% | 86.45% | 84.62% | 85.53% |
| BERT + Logistic Regression | 87.42% | 87.62% | 87.16% | 87.39% |

## Key Findings

- TF-IDF + Logistic Regression achieved the best overall performance.
- Word2Vec required less feature training time than BERT but achieved the lowest classification performance.
- BERT provided contextual word representations but required the highest computational time.
- TF-IDF demonstrated a good balance between classification performance and prediction speed.
- Validation and test results were consistent, particularly for TF-IDF.

## Conclusion

The experimental results show that **TF-IDF combined with Logistic Regression** was the most effective approach for this IMDB sentiment classification task. It achieved the highest test accuracy and F1-score while maintaining very fast prediction time.

Although BERT provides contextual language representations, its computational cost did not result in better performance in this frozen-embedding experiment. Overall, TF-IDF proved to be a practical and efficient solution for this dataset.
