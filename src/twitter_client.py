import re

import tweepy


class TwitterClient:
    def __init__(self, consumer_key, consumer_secret, ):
        auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
        self.api = tweepy.API(auth)

    def preprocess(self, tweet):
        tweet_text = tweet.full_text
        return ' '.join(re.sub("(RT @[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+: / {2}/ \S+)", " ", tweet_text).split())

    def query(self, search_query, tweets_limit=500):
        raw_tweets = tweepy.Cursor(self.api.search_tweets, q=search_query, lang="en", tweet_mode='extended').items(tweets_limit)
        processed_tweets = [self.preprocess(tweet) for tweet in raw_tweets]
        return processed_tweets

    def save(self, tweets, filename):
        if filename:
            with open(filename, "w") as sfile:
                sfile.write("\n".join(tweets))

class Tweet:
    def __init__(self, text, likes, comments, retweets, followers):
        self.text = text
        self.likes = likes
        self.comments = comments
        self.retweets = retweets
        self.followers = followers
        self.sentiment_score = None


def getTweets(symbol, name, industry):
    return [Tweet("good good", 0, 0, 0, 5), Tweet("bad badj", 0, 0, 0, 8)]


if __name__ == "__main__":
    twitter_client = TwitterClient("xlinnUvFlgoU2JQlynXU5Vx14",
                                   "36Pdrs0qNMqYUg8DBbVBRAiSC4i5yNb27Xo6Tm9XGg5JlNpBeT")
    tweets = twitter_client.query("$amzn")
    twitter_client.save(tweets, "tweets.txt")

