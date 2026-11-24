"""Simple Streamlit app for a demo movie-review sentiment classifier."""
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="Movie Review AI", page_icon="🎬")
st.title("🎬 Movie Review Sentiment AI")
st.write("Type a movie review and let the trained model predict its sentiment.")

@st.cache_resource
def load_model():
    model_file = Path(__file__).resolve().parent / "sentiment_model.joblib"
    return joblib.load(model_file)  # Only load trusted model files.

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

review = st.text_area(
    "Your movie review",
    value="The movie was amazing and the acting was excellent!",
    height=130,
    max_chars=3000,
)

if st.button("Analyse sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        # The saved pipeline handles TF-IDF and classification together.
        predicted_label = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]
        score_by_label = dict(zip(model.classes_, probabilities))
        confidence = score_by_label[predicted_label]

        if predicted_label == "positive":
            st.success("😊 Predicted sentiment: POSITIVE")
        else:
            st.error("😞 Predicted sentiment: NEGATIVE")

        st.write(f"**Model probability for predicted class:** {confidence:.1%}")
        st.progress(float(confidence))
        st.write("**Predicted class probabilities**")
        st.write(f"Positive: {score_by_label['positive']:.1%}")
        st.write(f"Negative: {score_by_label['negative']:.1%}")

st.caption(
    "Educational demonstration trained on a small mix of manually written and synthetic reviews. "
    "Predictions and confidence scores may be unreliable on real-world reviews."
)
