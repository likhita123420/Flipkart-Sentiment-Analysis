import streamlit as st
import pickle
import re
import nltk
import string
import contractions
import os
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# -----------------------------
# NLTK setup (run once)
# -----------------------------
@st.cache_resource
def setup_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

setup_nltk()

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    with open("best_sentiment_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# Text Cleaning Setup
# -----------------------------
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def clean(doc, stem=True):
    doc = contractions.fix(doc)
    doc = re.sub(r"[^a-zA-Z]", " ", doc)
    doc = doc.lower()
    tokens = word_tokenize(doc)
    tokens = [t for t in tokens if t not in stop_words]

    if stem:
        tokens = [stemmer.stem(t) for t in tokens]
    else:
        tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]

    return " ".join(tokens)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Flipkart Sentiment Analysis", layout="centered")

st.title("🛒 Flipkart Review Sentiment Analysis")
st.write("Enter a product review to predict whether it is **Positive or Negative**.")

review = st.text_area("✍️ Enter Review Text")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        clean_review = clean(review)
        prediction = model.predict([clean_review])[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")
