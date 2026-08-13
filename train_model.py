import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, multilabel_confusion_matrix, precision_recall_curve

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("Loading dataset...")
    dataset_path = r"C:\Users\HP\Desktop\My Machine learning Model\go_emotions_dataset.csv"
    df = pd.read_csv(dataset_path)
    
    # Filter out unclear examples
    if 'example_very_unclear' in df.columns:
        df = df[df['example_very_unclear'] == False]
    
    # Preprocess text
    print("Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Extract emotion columns (all columns after example_very_unclear, except id and text)
    emotion_cols = [col for col in df.columns if col not in ['id', 'text', 'example_very_unclear', 'clean_text']]
    
    print(f"Dataset shape after filtering: {df.shape}")
    print("Label distribution (number of positive examples per class):")
    print(df[emotion_cols].sum().sort_values(ascending=False))
    
    X = df['clean_text']
    y = df[emotion_cols]
    
    # Train-val-test split (70-15-15)
    # First split off 15% for test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    # Then split the remaining 85% into train (70%) and val (15%) -> 15/85 = 0.17647
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.17647, random_state=42)
    
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training model...")
    # Using LogisticRegression without class_weight='balanced'
    base_lr = LogisticRegression(random_state=42, max_iter=1000)
    model = OneVsRestClassifier(base_lr)
    
    model.fit(X_train_vec, y_train)
    
    print("Evaluating model with default 0.5 threshold on Test Set...")
    y_pred_test_default = model.predict(X_test_vec)
    
    print("\nClassification Report (Test Set - Threshold 0.5):")
    print(classification_report(y_test, y_pred_test_default, target_names=emotion_cols, zero_division=0))
    
    print("\nFinding optimal thresholds per class on Validation Set...")
    y_val_pred_proba = model.predict_proba(X_val_vec)
    optimal_thresholds = []
    
    for i, class_name in enumerate(emotion_cols):
        precision, recall, thresholds = precision_recall_curve(y_val.iloc[:, i], y_val_pred_proba[:, i])
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        opt_thresh = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_thresholds.append(opt_thresh)
        print(f"Optimal threshold for {class_name}: {opt_thresh:.4f}")
        
    print("\nEvaluating model with optimal thresholds on Held-Out Test Set...")
    y_test_pred_proba = model.predict_proba(X_test_vec)
    y_pred_test_opt = (y_test_pred_proba >= optimal_thresholds).astype(int)
    
    print("\nFinal Classification Report (Held-Out Test Set - Optimal Thresholds):")
    print(classification_report(y_test, y_pred_test_opt, target_names=emotion_cols, zero_division=0))
    
    # Save model and vectorizer
    print("Saving model and vectorizer...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(emotion_cols, 'models/class_names.joblib')
    joblib.dump(optimal_thresholds, 'models/optimal_thresholds.joblib')
    
    print("Training complete! Model saved in 'models/' directory.")

if __name__ == "__main__":
    main()
