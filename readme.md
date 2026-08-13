# 🎭 Text Emotion Classifier

A lightweight, multi-label text emotion classification web app built with **Scikit-learn** and **Streamlit**, trained on Google's **GoEmotions** dataset. The app predicts one or more emotions conveyed in a given piece of text, along with confidence scores.

## 🔍 Overview

This project classifies text into **28 fine-grained emotion categories** (e.g., admiration, anger, gratitude, sadness, neutral) using a classical machine learning pipeline — no deep learning or transformer models required. It demonstrates that a well-tuned TF-IDF + Logistic Regression pipeline can achieve solid performance on a challenging, imbalanced, multi-label emotion dataset.

## ✨ Features

- **Multi-label classification** — a single text can express multiple emotions simultaneously
- **Per-class optimal thresholds** — instead of a single fixed cutoff, each emotion has its own tuned probability threshold (maximized via F1-score on a validation set) for more accurate predictions
- **Negation-aware preprocessing** — custom text cleaning that merges negation words with the following token (e.g., _"not happy"_ → _"not_happy"_) to reduce misclassification caused by negated statements
- **Interactive Streamlit UI** — simple text input, one-click prediction, and a probability distribution bar chart
- **Proper train/validation/test split (70/15/15)** — thresholds are tuned on validation data and final metrics are reported on a completely held-out test set for honest evaluation

## 🧠 Model Details

| Component     | Details                                                                                                                     |
| ------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Dataset       | [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (~208K examples, 28 emotion labels) |
| Vectorization | TF-IDF (`max_features=15000`, `ngram_range=(1,2)`)                                                                          |
| Classifier    | `OneVsRestClassifier` wrapping `LogisticRegression`                                                                         |
| Thresholding  | Per-class optimal thresholds via precision-recall curve analysis                                                            |
| Evaluation    | Macro F1-score: **0.37** on held-out test set                                                                               |

## 📁 Project Structure

emotion_classifier/
├── app.py # Streamlit web application
├── train_model.py # Model training script
├── models/
│ ├── model.joblib # Trained OneVsRestClassifier
│ ├── vectorizer.joblib # Fitted TF-IDF vectorizer
│ ├── class_names.joblib # List of emotion labels
│ └── optimal_thresholds.joblib # Per-class prediction thresholds
└── README.md

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit pandas scikit-learn joblib matplotlib numpy
```

### Train the Model

```bash
python train_model.py
```

This preprocesses the dataset, trains the model, evaluates it, and saves all artifacts to the `models/` directory.

### Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## 📊 Example Predictions

| Input Text                            | Predicted Emotion(s)        |
| ------------------------------------- | --------------------------- |
| "Thank you so much for your help!"    | Gratitude (high confidence) |
| "I love spending time with my family" | Love                        |
| "That joke was hilarious"             | Amusement                   |

## ⚠️ Known Limitations

- Struggles with **sarcasm** and highly **idiomatic phrases**, since the model relies on literal word patterns rather than deeper contextual understanding
- Negation handling is improved via custom preprocessing but not perfect — complex sentence structures can still confuse the model
- Trained exclusively on **English** text; performance on other languages (e.g., Roman Urdu) is unreliable

## 🔮 Future Improvements

- Fine-tune a transformer-based model (e.g., DistilBERT) for better contextual and negation understanding
- Expand negation handling to cover more complex grammatical patterns
- Add multi-language support

## 🛠️ Built With

- [Python](https://www.python.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)

## 📄 Dataset Credit

This project uses the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions) released by Google Research.
