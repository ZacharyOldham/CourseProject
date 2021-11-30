import os

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def relevance_judgement_using_textblob(tweets_file, output_file):
    tweets = load_tweets(tweets_file)

    tweets_with_sentiment = []
    for tweet in tweets:
        testimonial = TextBlob(tweet)
        # print(testimonial.sentiment)
        tweets_with_sentiment.append(f"{tweet} -> {testimonial.sentiment.polarity}, {testimonial.sentiment.subjectivity} ")
    # print(tweets_with_sentiment)

    save_output(tweets_with_sentiment, output_file)


def relevance_judgement_using_vader(tweets_file, output_file):
    tweets_with_sentiment = []
    tweets = load_tweets(tweets_file)
    for tweet in tweets:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(tweet)
        # print("{} {}".format(tweet, str(scores)))
        tweets_with_sentiment.append(f"{tweet} -> {scores['pos']}, {scores['neg']}, {scores['neu']} ")

    # print(tweets_with_sentiment)
    save_output(tweets_with_sentiment, output_file)


def load_tweets(tweets_file):
    with open(tweets_file) as fp:
        lines = fp.readlines()
    return lines


def save_output(list_with_score, outfile):
    with open(outfile, "w") as ofile:
        ofile.write("\n".join([line for line in list_with_score]))


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.curdir)
    print(ROOT_DIR)
    relevance_judgement_using_textblob("../relevance/raw_tweets.txt", "../relevance/relevance_textblob.txt")
    relevance_judgement_using_vader("../relevance/raw_tweets.txt", "../relevance/relevance_vader.txt")
