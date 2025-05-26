from sentence_transformers import SentenceTransformer, util
from collections import Counter
import nltk
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s#@]", "", text)
    return text.strip().lower()

def extract_keywords(texts, top_k=10):
    all_words = []
    for text in texts:
        clean = clean_text(text)
        all_words.extend(clean.split())
    common = Counter(all_words).most_common(top_k)
    return [word for word, _ in common if len(word) > 3]

def get_text_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)
