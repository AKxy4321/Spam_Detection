import os
import re
import string
import sys
import types

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from PyPDF2 import PdfReader

# Ensure NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Initialize NLP tools
stopwords_set = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


# ----------------- Text processing -----------------
def clean_text_for_spam(text: str) -> str:
    """Normalize whitespace and remove excessive line breaks."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_spam(text: str, use_stem: bool = True) -> list:
    """Tokenize text and optionally stem or lemmatize."""
    if not text:
        return []

    text = text.lower()
    tokens = nltk.word_tokenize(text)
    cleaned = []

    for token in tokens:
        token = token.strip(string.punctuation)
        if len(token) <= 1:
            continue
        token = stemmer.stem(token) if use_stem else lemmatizer.lemmatize(token)
        cleaned.append(token)

    return cleaned


module_name = "__main__"
module = types.ModuleType(module_name)
sys.modules[module_name] = module

# ----------------- Load spam pipeline -----------------
spam_pipeline_path = os.path.join(os.path.dirname(__file__), "spam_pipeline.pkl")


class SpamPipeline:
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model

    def preprocess(self, texts: list) -> list:
        return [
            " ".join(tokenize_for_spam(clean_text_for_spam(text))) for text in texts
        ]

    def predict(self, texts: list) -> list:
        processed = self.preprocess(texts)
        X = self.vectorizer.transform(processed)
        return self.model.predict(X)


setattr(module, "SpamPipeline", SpamPipeline)

# Load once for reuse
pipeline_nb = joblib.load(spam_pipeline_path)


# ----------------- PDF prediction -----------------
def predict_pdf(path: str, label_map={0: "HAM", 1: "SPAM"}) -> dict:
    """
    Extract text from PDF, preprocess, vectorize, and predict spam/ham.

    Returns a dict with prediction info.
    """
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            break  # only first page
    except Exception as e:
        return {"error": f"Failed to read PDF: {e}"}

    processed_text = " ".join(pipeline_nb.preprocess([text])[0].split())
    if not processed_text.strip():
        return {"error": "No valid text extracted from PDF"}

    pred = pipeline_nb.predict([processed_text])[0]

    return {
        "prediction": int(pred),
        "label": label_map[pred],
        "text_length": len(processed_text),
        "preview": processed_text[:200] + ("..." if len(processed_text) > 200 else ""),
    }
