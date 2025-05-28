from analysis.text_analysis import extract_keywords, get_text_embeddings, clean_tweet, extract_topics, extract_hashtags, analyze_tone

def build_persona_from_text(tweets, recency_weight=0.75):
    contents = [t['content'] for t in tweets]
    keywords = extract_keywords(contents)
    embeddings = get_text_embeddings(contents)

    return {
        "keywords": keywords,
        "text_embedding": embeddings.mean(axis=0).tolist()
    }
    
def build_persona(tweets):
    contents = [clean_tweet(t['content']) for t in tweets]
    topics = extract_topics(contents, top_k=10)
    hashtags = extract_hashtags([t['content'] for t in tweets], top_k=5)
    tone = analyze_tone(contents)
    sample_tweets = contents[:5]

    return {
        "interests": topics[:5],
        "topics": hashtags,
        "vocabulary_style": "witty and concise",  # optionally LLM-generated
        "tone": tone,
        "audience": "tech enthusiasts and founders",  # optional manual/LLM
        "values": ["innovation", "freedom", "self-growth"],  # optional manual/LLM
        "representative_tweets": sample_tweets
    }

