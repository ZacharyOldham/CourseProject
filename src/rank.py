from collections import defaultdict
from gensim import corpora
from gensim.summarization import bm25
import nltk
from nltk.corpus import stopwords
import os

nltk.download("stopwords", quiet=True)


# Class for ranking of tweets according to query and additional processing.

class TweetRanking:

    def __init__(self, corpus, query="stock market", output_file="output.txt", top_k=20, remove_stopwords=True,
                 remove_digits=True, remove_rare_words=True, write_to_file=False):
        """
        Rank provided tweets using BM25 scoring.

        :param corpus: Set of tweets to be ranked
        :param query: Query to be used to rank tweets in corpus
        :param output_file: file to save ranked top_k tweets
        :param top_k: number of ranked tweets to be returned and saved to output_file
        :param remove_stopwords: Option to remove stopwords from set of tweets, default = True
        :param remove_digits: Option to remove numbers from set of tweets, default = True
        :param remove_rare_words: Option to remove word that appear only once, default = True
        BM25 Parameters:
        PARAM_K1 = 1.5
        PARAM_B = 0.75
        EPSILON = 0.25
        """
        self.original_tweets = []
        self.corpus = corpus
        self.query = query
        self.output_file = output_file
        self.top_k = top_k
        self.remove_stopwords = remove_stopwords
        self.remove_digits = remove_digits
        self.remove_rare_words = remove_rare_words
        self.write_to_file = write_to_file
        self.preprocess_file = "preprocess.txt"

    def save_to_file(self, tweet, file):
        with open(file, "a") as output:
            output.write(" ".join(tweet))
            output.write("\n")
            output.close()

    # Preprocess of tweets, remove stop words and numbers from tweets if options are enabled.

    def preprocess_tweet(self, tweet):
        process_tweet = [word.lower() for word in tweet.split()]

        if self.remove_stopwords:
            stop_words = stopwords.words()
            process_tweet = [word for word in process_tweet if word not in stop_words]
        if self.remove_digits:
            process_tweet = [word for word in process_tweet if word.isalpha()]
        return process_tweet

    # Create file with selected Tweets

    def pre_process_corpus(self):
        open(self.preprocess_file, "w+")
        open(self.preprocess_file, "r+").truncate(0)
        with open(self.corpus, "r") as tweets_doc:
            for tweet in tweets_doc:
                tweet = tweet.strip()
                self.original_tweets.append(tweet)
                self.save_to_file(self.preprocess_tweet(tweet), self.preprocess_file)

    # Count words frequencies in corpus for processing of rare words.

    def count_words(self):
        frequency = defaultdict(int)
        with open(self.preprocess_file, "r") as tweets:
            for tweet in tweets:
                for word in tweet.split():
                    frequency[word] += 1
        tweets.close()
        return frequency

    # Load processed tweets and remove rare words if option is enabled.

    def get_processed_corpus(self):
        frequency_count = []
        output = []

        if self.remove_rare_words:
            frequency_count = self.count_words()

        with open(self.preprocess_file, "r") as tweets:
            # Remove rare words with frequency equal to 1
            if self.remove_rare_words:
                processed_tweets = [[word for word in tweet.split() if frequency_count[word] > 1]
                                    for tweet in tweets]
            else:
                processed_tweets = [[word for word in tweet.split()]
                                    for tweet in tweets]
            for tweet in processed_tweets:
                if tweet not in output:
                    output.append(tweet)

        tweets.close()

        return output

    def get_ranked_documents(self):
        """
        Rank documents according to provided query using Gensim BM25 implementation,
        also calls tweets pre processing method.

        :return: Object with top_k ranked tweets, also writes file with the ranked tweets
        """
        # Preprocess tweets
        self.pre_process_corpus()
        # Load tweets and remove stop words if option is enabled
        processed_corpus = self.get_processed_corpus()
        # Creates a vocabulary from the list of tweets.
        dictionary = corpora.Dictionary(processed_corpus)
        # Converts each tweet to a BOW representation, a list of tuples with the term id and count.
        corpus = [dictionary.doc2bow(text) for text in processed_corpus]
        # Implementation of the BM25 ranking function.
        bm25_obj = bm25.BM25(corpus)
        # Converts query to BOW representation.
        query_doc = dictionary.doc2bow(self.query.split())
        # Return the relevance scores of all the tweets in relation to the query.
        scores = bm25_obj.get_scores(query_doc)
        # Sort and Select the top_k scored tweets and return them.
        best_tweets = sorted(range(len(scores)), key=lambda i: scores[i])[-self.top_k:]
        if self.write_to_file:
            open(self.output_file, "w+")
            open(self.output_file, "r+").truncate(0)
        output = []
        for tweet_index in best_tweets:
            if self.write_to_file:
                self.save_to_file(self.original_tweets[tweet_index].split(" "), self.output_file)
            output.append(tweet_index)

        os.remove(self.preprocess_file)
        return output


if __name__ == "__main__":
    ranker = TweetRanking("tweets.txt", "amazon amzn stock market value verge", "ranked_tweets.txt", 25)

    [print(tweet) for tweet in ranker.get_ranked_documents()]
