import csv
import math
import twitter_client
import rank
import sentiment_analysis


class Stocks:
    def __init__(self):
        self.stocks = []
        self.symbol_lookup = {}
        self.name_lookup = {}

    def addStock(self, symbol, name, industry):
        self.stocks.append((symbol, name, industry))
        self.symbol_lookup[symbol] = len(self.stocks) - 1
        self.name_lookup[name] = len(self.stocks) - 1

    def getSymbolFromName(self, name):
        return self.stocks[self.name_lookup[name]][0]

    def getIndustryFromName(self, name):
        return self.stocks[self.name_lookup[name]][2]

    def getNameFromSymbol(self, symbol):
        return self.stocks[self.symbol_lookup[symbol]][1]

    def getIndustryFromSymbol(self, symbol):
        return self.stocks[self.symbol_lookup[symbol]][2]

    def isSymbol(self, symbol):
        return symbol in self.symbol_lookup

    def isName(self, name):
        return name in self.name_lookup

def computeSentimentScore(positive_tweets, negative_tweets):
    pos_score = 0
    neg_score = 0
    for tweet in positive_tweets:
        pos_score += ((tweet.likes + 1.0) ** (1.0 / 3.0)) * ((tweet.comments + 1.0) ** (1.0 / 2.0)) * ((tweet.retweets + 1.0) ** (1.0 / 3.0)) * ((tweet.followers + 1.0) ** (1.0 / 4.0))

    for tweet in negative_tweets:
        neg_score += ((tweet.likes + 1.0) ** (1.0 / 3.0)) * ((tweet.comments + 1.0) ** (1.0 / 2.0)) * ((tweet.retweets + 1.0) ** (1.0 / 3.0)) * ((tweet.followers + 1.0) ** (1.0 / 4.0))

    if pos_score + neg_score == 0:
        return 0

    return (pos_score - neg_score) / (pos_score + neg_score)

if __name__ == "__main__":

    # Load stock informtion
    stocks = Stocks()
    with open("../Data/stocks.csv", "r") as f:
        reader = csv.reader(f)
        stock_list = list(reader)
        for stock in stock_list:
            stocks.addStock(stock[0].lower(), stock[1].lower(), stock[3])

    print("Enter stock ticker or company name. Company name must be exact.")
    symbol = None
    name = None
    industry = None
    while symbol is None:
        user_input = input()
        clean_input = user_input.strip().lower()
        if stocks.isName(clean_input):
            name = clean_input
            symbol = stocks.getSymbolFromName(name)
            industry = stocks.getIndustryFromName(name)
        elif stocks.isSymbol(clean_input):
            symbol = clean_input
            name = stocks.getNameFromSymbol(symbol)
            industry = stocks.getIndustryFromSymbol(symbol)
        else:
            print("Invalid input: \"" + user_input + "\" is not a valid symbol or company name.")

    # Get and update the run number
    run_rumber = None
    with open("../Tweets/run_number", "r") as f:
        run_number = int(f.read())
    run_number += 1
    with open("../Tweets/run_number", "w") as f:
        f.write(str(run_number))
    print("Outputs stored with run number " + str(run_number))

    # Retrieve relevant-ish tweets from twitter
    twitter_client = twitter_client.TwitterClient()
    tweets = twitter_client.get_tweets(symbol, name, industry)
    twitter_client.save(tweets, "asdf.txt")
    tweets_lookup = {}
    for i in range(0, len(tweets)):
        tweet = tweets[i]
        tweets_lookup[tweet.text] = i

    # Save all tweets to a file so we can find the best ones
    all_tweet_file = "../Tweets/all_tweets_" + str(run_number) + ".txt"
    ranked_tweet_file = "../Tweets/relevant_tweets_" + str(run_number) + ".txt"
    positive_tweet_file = "../Tweets/positive_tweets_" + str(run_number) + ".txt"
    negative_tweet_file = "../Tweets/negative_tweets_" + str(run_number) + ".txt"

    with open(all_tweet_file, "w") as f:
        for tweet in tweets:
            f.write(tweet.text + "\n")

    # # Rank tweets, get best 50 tweets
    query = symbol + " " + name
    ranker = rank.TweetRanking(all_tweet_file, query, ranked_tweet_file, 50, write_to_file=True)
    best_tweets_text = ranker.get_ranked_documents()
    best_tweets = []
    for i in range(0, len(best_tweets_text)):
        best_tweets.append(tweets[tweets_lookup[best_tweets_text[i]]])
    
    # Classfiy the best tweets (0 = negative, 1 = positive)
    model, idx2word, word2idx = sentiment_analysis.build_model()
    # print("DONE BUILDING MODEL")
    # print(best_tweets_text)
    labels = sentiment_analysis.predict(model, idx2word, word2idx, best_tweets_text)
    for i in range(0, len(best_tweets)):
        best_tweets[i].sentiment_score = labels[i]

    positive_tweets = []
    negative_tweets = []
    for tweet in best_tweets:
        if tweet.sentiment_score == 0:
            negative_tweets.append(tweet)
        else:
            positive_tweets.append(tweet)

    positive_tweets.sort(key=lambda x: x.sentiment_score, reverse=True)
    negative_tweets.sort(key=lambda x: x.sentiment_score)

    # Persist the classifications
    with open(positive_tweet_file, "w") as f:
        for tweet in positive_tweets:
            f.write(tweet.text)
    
    with open(negative_tweet_file, "w") as f:
        for tweet in negative_tweets:
            f.write(tweet.text)

    # Compute sentiment score
    score = computeSentimentScore(positive_tweets, negative_tweets)
    print("Positive Tweets: ")
    print([x.text for x in positive_tweets])
    print("Negative Tweets:")
    print([x.text for x in negative_tweets])
    print("Sentiment score (ranges from -1 to 1): " + str(score))