# Counterfeit Job Posting Detection using NLP & SMOTE

## Overview
Automated detection of fraudulent job postings using Natural Language Processing (NLP) techniques and Machine Learning classifiers. The dataset contains ~17,880 postings with severe class imbalance (~95% real vs ~5% fake).

## Key Features & Techniques
- **Preprocessing:** Combined multiple text fields (`title`, `company_profile`, `description`, `requirements`, `benefits`) and cleaned missing data.
- **Feature Extraction:** TF-IDF Vectorization with unigrams and bigrams (`ngram_range=(1,2)`).
- **Class Imbalance Handling:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance minority class representations in the training set.
- **Classification:** Trained a **Random Forest Classifier** (`n_estimators=100`).

## Results
- **ROC-AUC Score:** 0.99
- **Precision (Fake Class):** 0.99
- **False Positive Count:** 1 out of 3,403 real job postings.

## Environment & Libraries
`Python`, `scikit-learn`, `imblearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

**Added evaluation plots:** Confusion Matrix and ROC Curve
