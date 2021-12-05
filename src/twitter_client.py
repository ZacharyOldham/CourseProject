import re
from dataclasses import dataclass

import tweepy


@dataclass
class Tweet:
    """
    Data class for holding processed tweets. Each tweet will be stored as an object

    text:
        Tweet text (preprocessed text)
    likes_count:
        Total number of likes for the tweet
    retweets_count:
        Total number of times the tweet was retweeted.
    followers_count:
        Total number of followers of the user who posted the tweet
    sentiment_score:
        Sentiment score of the tweet. This is computed by the ml model
    """
    text: str = ""
    likes_count: int = 0
    retweets_count: int = 0
    followers_count: int = 0
    sentiment_score: float = 0.0


class TwitterClient:
    """
    Twitter client for loading tweets. Take consumer key and secret as parameters which are created from the
    twitter developer account
    """

    def __init__(self, consumer_key="xlinnUvFlgoU2JQlynXU5Vx14",
                 consumer_secret="36Pdrs0qNMqYUg8DBbVBRAiSC4i5yNb27Xo6Tm9XGg5JlNpBeT"):
        auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
        self.api = tweepy.API(auth)
        self.tickers_threshold = 3

    def get_tweets(self, symbol, name="", industry="",
                   allow_duplicates=False, tweets_limit=500, tickers_threshold=3):
        """
        Loads tweets from twitter using the give parameters

        :param symbol:
            Stock symbol (e.g: AMZN)
        :param name:
            Name of the company (optional). Currently not used
        :param industry:
            Company Industry (optional). Currently not used
        :param allow_duplicates:
            Allows duplicate tweets if set to true. Default is false
        :param tweets_limit:
            Total number of tweets to load. Default is 500.
            This is the raw tweets counts. Final processed tweets count would be less than this number
        :param tickers_threshold:
            Maximum number of tickers allowed in the tweet
        :return:
        """
        self.tickers_threshold = tickers_threshold
        all_tweets = self.__query(f"${symbol}", tweets_limit)
        tweets = []
        if not allow_duplicates:
            text_list = []
            for tweet in all_tweets:
                if tweet.text not in text_list:
                    text_list.append(tweet.text)
                    tweets.append(tweet)
        else:
            tweets = all_tweets
        return tweets

    def __query(self, search_query, tweets_limit=50):
        # Queries twitter using tweepy api.
        # We are loading most recent tweets and setting a limit on the number of tweets to load.
        raw_tweets = tweepy.Cursor(self.api.search_tweets,
                                   q=search_query, lang="en",
                                   result_type="recent",
                                   tweet_mode="extended").items(tweets_limit)

        # Tweets that are dropped in preprocessing will result in a Tweet objects with empty text.
        # We are filtering those blank tweet objects here
        processed_tweets = []
        for rt in raw_tweets:
            try:
                if rt.retweeted_status:
                    pt = self.__preprocess(rt.retweeted_status)
                else:
                    pt = self.__preprocess(rt)
            except AttributeError:
                pt = self.__preprocess(rt)
            if len(pt.text) > 0:
                processed_tweets.append(pt)

        return processed_tweets

    def __preprocess(self, raw_tweet):
        #
        # Tweets with more than 3 stock tickers doesn't give good details about the specific stock in the query.
        # We drop such tweets to have a clean data.
        #
        tickers_in_tweet = set(re.compile('\$\w+').findall(raw_tweet.full_text))
        if len(tickers_in_tweet) > self.tickers_threshold:
            # print(tickers_in_tweet)
            # print(f"tweet dropped -> {raw_tweet}")
            return Tweet()

        # Remove emojis and any other non-ascii characters
        ASCII = ''.join(chr(x) for x in range(128))
        raw_str = raw_tweet.full_text
        demoji_str = ""
        for c in raw_str:
            if c in ASCII:
                demoji_str += c

        # Remove newlines
        demoji_str = demoji_str.strip().replace("\n", " ")

        # Force lowercase
        demoji_str = demoji_str.lower()

        # Tweet object with the processed tweet and other meta data
        tweet = Tweet()
        tweet.text = demoji_str
        tweet.likes_count = raw_tweet.favorite_count
        tweet.retweet_count = raw_tweet.retweet_count
        tweet.followers_count = raw_tweet.user.followers_count
        return tweet

    def save(self, tweets, filename):
        """
        Helper function to save the tweets to a file.
        :param tweets:
            List of Tweet objects
        :param filename:
            Output file name with the full path
        """
        if filename:
            with open(filename, "w") as sfile:
                sfile.write("\n".join([tweet.text for tweet in tweets]))


if __name__ == "__main__":
    twitter_client = TwitterClient()
    loaded_tweets = twitter_client.get_tweets("amzn", allow_duplicates=False, tweets_limit=10, tickers_threshold=3)
    # print(loaded_tweets)
    # twitter_client.save(loaded_tweets, "tweets.txt")
