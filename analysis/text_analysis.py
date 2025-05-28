from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer

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


def analyze_tone(texts):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)['compound'] for t in texts]
    avg_score = sum(scores) / len(scores)
    if avg_score > 0.3:
        return "positive"
    elif avg_score < -0.3:
        return "negative"
    else:
        return "neutral"

def extract_hashtags(texts, top_k=5):
    hashtags = []
    for t in texts:
        hashtags.extend(re.findall(r"#(\w+)", t.lower()))
    return [tag for tag, _ in Counter(hashtags).most_common(top_k)]


def extract_topics(texts, top_k=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_k)
    X = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out().tolist()

def clean_tweet(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()
