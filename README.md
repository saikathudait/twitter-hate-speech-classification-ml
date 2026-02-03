# MACHINE LEARNING-BASED CLASSIFICATION OF HATE SPEECH, OFFENSIVE LANGUAGE, AND NEUTRAL CONTENT ON TWITTER

## Project Overview
This project implements a machine learning system to classify Twitter text into **hate speech**, **offensive language**, and **neutral content**. The study addresses challenges posed by noisy social media data and class imbalance by applying robust preprocessing, feature engineering, and model optimisation techniques.

The solution follows a complete machine learning lifecycle from data exploration to deployment-ready prediction.

---

## Dataset
- **Name:** Hate Speech and Offensive Language Dataset  
- **Source:** CrowdFlower  
- **Total Tweets:** 24,783  
- **Final Samples Used:** 22,533 (after outlier removal)

### Class Labels
- `0` – Hate Speech  
- `1` – Offensive Language  
- `2` – Neither  

### Dataset File
```text
labeled_data.csv
```

### Data Preprocessing

Converted all text to lowercase

Removed URLs, mentions, hashtags, numbers, punctuation, and HTML entities

Removed extremely short and long tweets using percentile-based outlier detection

Created an additional numeric feature: tweet length

### Feature Engineering

TF-IDF Vectorisation

Unigrams and bigrams

Maximum vocabulary size: 20,000

Dimensionality Reduction

TruncatedSVD with 200 components

Numeric Feature

Scaled tweet length using StandardScaler

Final feature dimension: 201 features per tweet

### Machine Learning Models

The following models were trained and evaluated:

Logistic Regression

Linear Support Vector Machine (SVM)

Multinomial Naive Bayes

Random Forest

### Model Evaluation Metrics

Accuracy

Precision (Macro and Weighted)

Recall (Macro and Weighted)

F1-score

Confusion Matrix

Log Loss

Multi-class ROC-AUC (One-vs-Rest)

### Best Performing Model

Tuned Logistic Regression achieved the best balance across all metrics.

Accuracy: ~88%

Macro F1-score: ~0.65

ROC-AUC (macro): ~0.90

Log Loss: ~0.32

This model was selected due to its strong generalisation performance, probability outputs, and interpretability.

### Model Deployment

A deployment pipeline was implemented to classify new tweets using the trained model.
The pipeline applies identical preprocessing, feature extraction, scaling, and prediction steps as used during training.

Saved Model Files

best_logistic_regression_model.pkl
tfidf_vectorizer.pkl
svd_transformer.pkl
length_scaler.pkl

### Repository Structure
twitter-hate-speech-classification-ml/
│
├── labeled_data.csv
├── machine_learning_twitter_classification.py
├── README.md
├── best_logistic_regression_model.pkl
├── tfidf_vectorizer.pkl
├── svd_transformer.pkl
└── length_scaler.pkl


### Key Findings

Class imbalance significantly affects hate speech recall

Offensive language is detected with high accuracy due to abundant samples

Dimensionality reduction improves computational efficiency and stability

Logistic Regression provides the best trade-off between performance and interpretability

### Future Enhancements

Apply oversampling techniques such as SMOTE

Introduce class-weighted loss functions

Explore transformer-based models like BERT

Expand the dataset for improved minority-class learning
