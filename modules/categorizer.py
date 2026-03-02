"""
Module 1: NLP-Based Expense Categorization
- TF-IDF + Logistic Regression (fast & strong baseline)
- Text cleaning and feature engineering
- Evaluation with accuracy, F1, confusion matrix
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


def clean_text(text: str) -> str:
    """Clean transaction description for NLP processing"""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)           # remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)      # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # normalize whitespace
    # Remove common noise words
    noise_words = ['payment', 'transfer', 'neft', 'upi', 'imps', 'pos', 'bill', 'order']
    words = text.split()
    words = [w for w in words if w not in noise_words and len(w) > 2]
    return ' '.join(words)


def build_categorizer(df: pd.DataFrame):
    """Train TF-IDF + Logistic Regression categorizer"""
    df = df.copy()
    df['clean_description'] = df['description'].apply(clean_text)
    
    X = df['clean_description']
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=1
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=5.0,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    results = {
        'model': pipeline,
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classes': pipeline.classes_,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return results


def predict_category(model, description: str) -> dict:
    """Predict category for a single transaction"""
    clean = clean_text(description)
    pred = model.predict([clean])[0]
    proba = model.predict_proba([clean])[0]
    classes = model.classes_
    
    return {
        'predicted_category': pred,
        'confidence': round(max(proba) * 100, 1),
        'probabilities': {cls: round(p * 100, 1) for cls, p in zip(classes, proba)}
    }


def add_predicted_categories(df: pd.DataFrame, model) -> pd.DataFrame:
    """Add AI-predicted categories to dataframe"""
    df = df.copy()
    df['clean_description'] = df['description'].apply(clean_text)
    df['ai_category'] = model.predict(df['clean_description'])
    
    probas = model.predict_proba(df['clean_description'])
    df['confidence'] = (probas.max(axis=1) * 100).round(1)
    
    return df


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data.generate_data import generate_transactions
    
    df = generate_transactions()
    print(f"Training on {len(df)} transactions...")
    
    results = build_categorizer(df)
    print(f"\n✅ Model Accuracy: {results['accuracy']:.2%}")
    print("\n📊 Classification Report:")
    report = results['classification_report']
    for cat, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"  {cat}: F1={metrics['f1-score']:.2f}, Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}")
    
    # Test predictions
    test_cases = [
        "Swiggy Order",
        "Uber Ride",
        "Netflix Monthly",
        "House Rent Transfer",
        "PVR Cinema Tickets"
    ]
    
    print("\n🔍 Sample Predictions:")
    for desc in test_cases:
        result = predict_category(results['model'], desc)
        print(f"  '{desc}' → {result['predicted_category']} ({result['confidence']}% confident)")
