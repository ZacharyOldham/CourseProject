from collections import defaultdict
from gensim import corpora
from gensim.summarization import bm25
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")


class TweetRanking:

    def __init__(self, corpus, query="stock market", output_file="output.txt", top_k=20, remove_stopwords=True,
                 remove_digits=True, remove_rare_words=True):
        """
        Rank provided tweets using BM25 scoring.

        :param corpus: Set of tweets to be ranked
        :param query: Query to be used to rank tweets in corpus
        :param output_file: file to save ranked top_k tweets
        :param top_k: number of ranked tweets to be returned and saved to output_file
        :param remove_stopwords: Option to remove stopwords from set of tweets, default = True
        :param remove_digits: Option to remove numbers from set of tweets, default = True
        :param remove_rare_words: Option to remove word that appear only once, default = True
        """
        self.corpus = corpus
        self.query = query
        self.output_file = output_file
        self.top_k = top_k
        self.remove_stopwords = remove_stopwords
        self.remove_digits = remove_digits
        self.remove_rare_words = remove_rare_words
        self.preprocess_file = "preprocess.txt"

    def save_to_file(self, tweet, file):
        with open(file, "a") as output:
            output.write(" ".join(tweet))
            output.write("\n")
            output.close()

    def preprocess_tweet(self, tweet):
        process_tweet = [word.lower() for word in tweet.split()]

        if self.remove_stopwords:
            stop_words = stopwords.words()
            process_tweet = [word for word in process_tweet if word not in stop_words]
        if self.remove_digits:
            process_tweet = [word for word in process_tweet if word.isalpha()]
        return process_tweet

    def pre_process_corpus(self):
        open(self.preprocess_file, "w+")
        open(self.preprocess_file, "r+").truncate(0)
        with open(self.corpus, "r") as tweets_doc:
            for tweet in tweets_doc:
                self.save_to_file(self.preprocess_tweet(tweet), self.preprocess_file)

    def count_words(self):
        frequency = defaultdict(int)
        with open(self.preprocess_file, "r") as tweets:
            for tweet in tweets:
                for word in tweet.split():
                    frequency[word] += 1
        tweets.close()
        return frequency

    def get_processed_corpus(self):
        frequency_count = []
        output = []

        if self.remove_rare_words:
            frequency_count = self.count_words()

        with open(self.preprocess_file, "r") as tweets:
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
        self.pre_process_corpus()
        processed_corpus = self.get_processed_corpus()
        dictionary = corpora.Dictionary(processed_corpus)
        corpus = [dictionary.doc2bow(text) for text in processed_corpus]
        bm25_obj = bm25.BM25(corpus)
        query_doc = dictionary.doc2bow(self.query.split())
        scores = bm25_obj.get_scores(query_doc)
        best_tweets = sorted(range(len(scores)), key=lambda i: scores[i])[-self.top_k:]
        open(self.output_file, "w+")
        open(self.output_file, "r+").truncate(0)
        output = []
        for tweet_index in best_tweets:
            self.save_to_file(processed_corpus[tweet_index], self.output_file)
            output.append(processed_corpus[tweet_index])

        return output


if __name__ == "__main__":
    ranker = TweetRanking("tweets.txt", "amazon amzn stock market value verge", "ranked_tweets.txt", 25)

    [print(tweet) for tweet in ranker.get_ranked_documents()]
