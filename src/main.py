import csv
import math
import twitter_client
import rank
import sentiment_analysis
import statistics

# Class that maintains information about stock symbols, company names, and company industries
class Stocks:

    # Initialize the class with empty stocks
    def __init__(self):
        self.stocks = []
        self.symbol_lookup = {}
        self.name_lookup = {}

    # Add a stock
    # symbol: the symbol for the stock
    # name: the name of the company
    # industry: the industry the company is in
    def addStock(self, symbol, name, industry):
        self.stocks.append((symbol, name, industry))
        self.symbol_lookup[symbol] = len(self.stocks) - 1
        self.name_lookup[name] = len(self.stocks) - 1

    # get the stock symbol of a company
    # name: the name of the company
    def getSymbolFromName(self, name):
        return self.stocks[self.name_lookup[name]][0]

    # get the industry of a company
    # name: the name of the company
    def getIndustryFromName(self, name):
        return self.stocks[self.name_lookup[name]][2]

    # get the name of a company
    # symbol: the stock symbol of the company
    def getNameFromSymbol(self, symbol):
        return self.stocks[self.symbol_lookup[symbol]][1]

    # get the industry of a company
    # symbol: the stock symbol of the company
    def getIndustryFromSymbol(self, symbol):
        return self.stocks[self.symbol_lookup[symbol]][2]

    # check if a string is a valid stock symbol
    # symbol: the string to check
    def isSymbol(self, symbol):
        return symbol in self.symbol_lookup

    # check if a string is a valid company name
    # name: the string to check
    def isName(self, name):
        return name in self.name_lookup

# Compute a score that summarizes the general sentiment of the stock by generating a weight for positive, negative, and neutral tweets
# positive_tweets: a list of positive tweets
# negative_tweets: a list of negative tweets
# neutral_tweets: a list of neutral tweets
# neutral_cutoff: the cutoff used to determine if tweets are neutral
def computeSentimentScore(positive_tweets, negative_tweets, neutral_tweets, neutral_cutoff):
    pos_score = 0
    neg_score = 0
    neu_score = 0

    # Compute positive score
    for tweet in positive_tweets:
        if tweet.sentiment_score < 0:
            print("ERROR: NOT A POSITIVE TWEET")
            exit()
        pos_score += tweet.sentiment_score * ((tweet.likes_count + 1.0) ** (1.0 / 3.0)) * ((tweet.retweets_count + 1.0) ** (1.0 / 3.0)) * ((tweet.followers_count + 1.0) ** (1.0 / 4.0))

    # Compute negative score
    for tweet in negative_tweets:
        if tweet.sentiment_score > 0:
            print("ERROR: NOT A NEGATIVE TWEET")
            exit()
        neg_score -= tweet.sentiment_score * ((tweet.likes_count + 1.0) ** (1.0 / 3.0)) * ((tweet.retweets_count + 1.0) ** (1.0 / 3.0)) * ((tweet.followers_count + 1.0) ** (1.0 / 4.0))

    # Compute neutral score
    for tweet in neutral_tweets:
        if abs(tweet.sentiment_score) > neutral_cutoff:
            print("ERROR: NOT A NEUTRAL TWEET")
            exit()
        neu_score += (neutral_cutoff - abs(tweet.sentiment_score)) * ((tweet.likes_count + 1.0) ** (1.0 / 3.0)) * ((tweet.retweets_count + 1.0) ** (1.0 / 3.0)) * ((tweet.followers_count + 1.0) ** (1.0 / 4.0))

    # Compute and return overall score
    if pos_score + neg_score + neu_score == 0:
        return 0
    else:
        return (pos_score - neg_score) / (pos_score + neg_score + neu_score)

# Entry point for the entire project
if __name__ == "__main__":

    # Load stock informtion
    stocks = Stocks()
    with open("../Data/stocks.csv", "r") as f:
        reader = csv.reader(f)
        stock_list = list(reader)
        for stock in stock_list:
            stocks.addStock(stock[0].lower(), stock[1].lower(), stock[3])

    # Obtain symbol, name, and industry based on user input
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
    tweets = twitter_client.get_tweets(symbol, name, industry, tweets_limit=1000)
    tweets_lookup = {}
    for i in range(0, len(tweets)):
        tweet = tweets[i]
        tweets_lookup[tweet.text] = i

    # Save all tweets to a file so we can find the best ones
    all_tweet_file = "../Tweets/all_tweets_" + str(run_number) + ".txt"
    ranked_tweet_file = "../Tweets/relevant_tweets_" + str(run_number) + ".txt"
    positive_tweet_file = "../Tweets/positive_tweets_" + str(run_number) + ".txt"
    negative_tweet_file = "../Tweets/negative_tweets_" + str(run_number) + ".txt"
    neutral_tweet_file = "../Tweets/neutral_tweets_" + str(run_number) + ".txt"

    with open(all_tweet_file, "w") as f:
        for tweet in tweets:
            f.write(tweet.text + "\n")

    # Rank tweets, get most relevant 25% of tweets
    query = symbol + " " + name
    ranker = rank.TweetRanking(all_tweet_file, query, ranked_tweet_file, int(len(tweets) / 4.0), write_to_file=True)
    best_tweets_index = ranker.get_ranked_documents()
    best_tweets = []
    best_tweets_text = []
    for i in range(0, len(best_tweets_index)):
        best_tweets.append(tweets[best_tweets_index[i]])
        best_tweets_text.append(tweets[best_tweets_index[i]].text)
    
    # Get tweet classifications < 0 = negative, > 0 = positive
    model, idx2word, word2idx = sentiment_analysis.build_model()
    labels = sentiment_analysis.predict(model, idx2word, word2idx, best_tweets_text)
    for i in range(0, len(best_tweets)):
        best_tweets[i].sentiment_score = labels[i]

    # Separate the tweets by classification
    amplitudes = []
    positive_tweets = []
    negative_tweets = []
    for tweet in best_tweets:
        amplitudes.append(abs(tweet.sentiment_score))
        if tweet.sentiment_score < 0:
            negative_tweets.append(tweet)
        else:
            positive_tweets.append(tweet)
    
    # Move tweets whose degree of sentiment is > one std below the average degree into a 'neutral' list
    amplitude_std = statistics.stdev(amplitudes)
    amplitude_avg = statistics.mean(amplitudes)
    neutral_cutoff = amplitude_avg - amplitude_std
    neutral_tweets = []
    for i in range(len(positive_tweets) - 1, -1, -1):
        tweet = positive_tweets[i]
        if abs(tweet.sentiment_score) < neutral_cutoff:
            positive_tweets.pop(i)
            neutral_tweets.append(tweet)
    for i in range(len(negative_tweets) - 1, -1, -1):
        tweet = negative_tweets[i]
        if abs(tweet.sentiment_score) < neutral_cutoff:
            negative_tweets.pop(i)
            neutral_tweets.append(tweet)

    # Sort each category of tweets so that the strongest examples are first
    positive_tweets.sort(key=lambda x: x.sentiment_score, reverse=True)
    negative_tweets.sort(key=lambda x: x.sentiment_score)
    neutral_tweets.sort(key=lambda x: abs(x.sentiment_score))

    # Persist the classifications
    with open(positive_tweet_file, "w") as f:
        for tweet in positive_tweets:
            f.write(tweet.text)
    
    with open(negative_tweet_file, "w") as f:
        for tweet in negative_tweets:
            f.write(tweet.text)

    with open(neutral_tweet_file, "w") as f:
        for tweet in neutral_tweets:
            f.write(tweet.text)

    # Compute sentiment score and output
    score = computeSentimentScore(positive_tweets, negative_tweets, neutral_tweets, neutral_cutoff)
    print("\n\nPositive Tweets: \n")
    for tweet in positive_tweets:
        print(tweet.text + "\n")
    print("\n\nNegative Tweets: \n")
    for tweet in negative_tweets:
        print(tweet.text + "\n")
    print("\n\nNeutral Tweets: \n")
    for tweet in neutral_tweets:
        print(tweet.text + "\n")
    print("Sentiment score (ranges from -1 to 1): " + str(score))
