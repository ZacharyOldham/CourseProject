import csv
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

    # Retrieve relevant-ish tweets from twitter
    tweets = twitter_client.getTweets(symbol, name, industry)
    tweets_lookup = {}
    for i in range(0, len(tweets)):
        tweet = tweets[i]
        tweets_lookup[tweet.text] = i

    # Save all tweets to a file so we can find the best ones
    all_tweet_file = "../Tweets/all_tweets_" + str(run_number) + ".txt"
    with open(all_tweet_file, "w") as f:
        for tweet in tweets:
            f.write(tweet.text)

    # # Rank tweets, get best 200 tweets
    # best_tweet_indices = rank.get
    

    

    
