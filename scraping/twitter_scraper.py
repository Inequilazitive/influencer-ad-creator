import snscrape.modules.twitter as sntwitter

def scrape_twitter(username, max_posts=50):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterUserScraper(username).get_items()):
        if i >= max_posts:
            break
        tweets.append({
            "content": tweet.content,
            "date": tweet.date.isoformat()
        })
    return tweets
