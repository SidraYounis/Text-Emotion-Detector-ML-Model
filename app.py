
import streamlit as st
import joblib
import re
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="Text Emotion Classifier",
    page_icon="🎭",
    layout="centered"
)

# Function to clean text (same as training)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_models():
    """Load model, vectorizer, and class names from disk."""
    model_dir = "models"
    vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    class_names = joblib.load(os.path.join(model_dir, 'class_names.joblib'))
    optimal_thresholds = joblib.load(os.path.join(model_dir, 'optimal_thresholds.joblib'))
    return vectorizer, model, class_names, optimal_thresholds

# Main UI
st.title("🎭 Text Emotion Classifier")
st.write("This app uses a Logistic Regression model trained on the GoEmotions dataset to predict the emotions conveyed in text.")

# Load models
try:
    vectorizer, model, class_names, optimal_thresholds = load_models()
except FileNotFoundError:
    st.error("Model files not found. Please run `train_model.py` first to generate the models.")
    st.stop()

# User input
user_input = st.text_area("Enter your text here:", height=150, placeholder="e.g., I'm so excited to try this new machine learning model!")

# (Removed threshold slider as we now use optimal per-class thresholds)

if st.button("Predict Emotion", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to predict.")
    else:
        with st.spinner("Analyzing text..."):
            # Preprocess text
            cleaned_text = clean_text(user_input)
            
            if not cleaned_text:
                st.warning("No valid text found after cleaning.")
            else:
                # Vectorize
                vec_text = vectorizer.transform([cleaned_text])
                
                # Predict probabilities
                probs = model.predict_proba(vec_text)[0]
                
                # Combine class names, probabilities, and thresholds
                emotion_probs = list(zip(class_names, probs, optimal_thresholds))
                
                # Filter by per-class threshold and sort by probability
                filtered_emotions = [(emo, p) for emo, p, thresh in emotion_probs if p >= thresh]
                filtered_emotions.sort(key=lambda x: x[1], reverse=True)
                
                if not filtered_emotions:
                    # If nothing passes threshold, show the top 1 anyway just to provide some result
                    st.info("No emotions passed their optimal thresholds. Here is the top prediction:")
                    top_emotion = max(emotion_probs, key=lambda x: x[1])
                    filtered_emotions = [(top_emotion[0], top_emotion[1])]
                
                # Display results
                st.subheader("Predictions")
                
                for emo, prob in filtered_emotions:
                    st.write(f"**{emo.capitalize()}**: {prob:.2%} confidence")
                
                # Chart
                st.subheader("Probability Distribution")
                df_chart = pd.DataFrame(filtered_emotions, columns=['Emotion', 'Probability']).set_index('Emotion')
                st.bar_chart(df_chart)

st.markdown("---")
st.caption("Powered by Streamlit and Scikit-Learn")

