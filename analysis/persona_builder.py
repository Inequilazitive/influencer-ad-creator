from analysis.text_analysis import extract_keywords, get_text_embeddings

def build_persona_from_text(tweets, recency_weight=0.75):
    contents = [t['content'] for t in tweets]
    keywords = extract_keywords(contents)
    embeddings = get_text_embeddings(contents)

    return {
        "keywords": keywords,
        "text_embedding": embeddings.mean(axis=0).tolist()
    }
