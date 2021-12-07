# Twitter Stock Sentiment Analysis

## Code Overview

This section is intended to provide a brief overview of the codebase. All code is well-commented, so those interested in a deeper understanding of the code should use this section to understand the high-level structure of the program, but should utilize the in-file comments for a more fine-grained understanding. 

### src/main.py

This file is the primary entry point for our program. 

At the top of the file is a `Stocks` class. This is a utility class for easily interacting with `Data/stocks.csv`, which is a list of every stock ticker and corresponding company name. This is used for validating user input, as well as bolstering the user input with additional information beyond just the stock ticker. 

Below this class is a function called `computeSentimentScore`, which takes our lists of categorized Tweets and computes a heuristic sentiment score by building a weight for each category of Tweet and taking the ratio of the difference between the positive and negative weights, and the cumulative weight. 

The entry point for the program is the line `if __name__ == "__main__":`. From here, this file handles the high-level interactions with our other utility files. It first populates the `Stocks` class, and uses it to obtain a valid user input. From there, it interfaces with `src/twitter_client.py` to obtain a list of Tweets from Twitter, and persists these. It next interfaces with `src/rank.py` to obtain the 25% most relevant Tweets to the user's input. Next, it interfaces with `sentiment_analysis.py` to obtain sentiment judgements for each Tweet, and divides the Tweets into positive, negative, and neutral categories. It persists these lists. Finally, it uses the `computeSentimentScore` function to get an overall sentiment score and outputs to the user. The sentiment score is also persisted.

### src/twitter_client.py

This file defines classes and methods for retrieving queries from Twitter. 
There are two classes defined in this file, `TwitterClient` and `Tweet`. 

`TwitterClient` takes "consumer_key" and "consumer_secret" as init parameters. These keys are generated from Twitter developer account. 
Initializing the client with these keys are required to query the tweets from twitter. `Tweepy` is used for retrieving the tweets 

`get_tweets` method is the main api method provided by `TwitterClient` for the other modules in the code to retrieve tweets from twitter. This method takes the following parameters 

* `symbol`: Stock symbol (e.g: AMZN)

* `name`: Name of the company (optional)

* `industry`: Company Industry (optional)

* `allow_duplicates`: Allows duplicate tweets if set to true. Default is false.

* `tweets_limit`: Total number of tweets to load. Default is 500. This is the raw tweets counts. Final processed tweets count would be less than this number.

* `tickers_threshold`: Maximum number of tickers allowed in the tweet

`__query` method is an internal method used by `get_tweets` to load the tweets from Twitter. This method uses tweepy to search tweets. 
It loads the most recent tweets and sets a limit on the number of tweets to load from Twitter using the value specified in the parameter. 
This method takes the following parameters

* `search_query` - Query to search tweets. It is just the stock ticker for our project.

* `tweets_limit` - Number of tweets to load from Twitter. Defaults to 500 from `get_tweets` method. 

`__preprocess` method is another internal method used by `__query`. Loaded tweets are preprocessed to remove non-ascii characters, new lines and converted to lowercase to keep the text uniform. 
Tweets are also filtered in this function by dropping any tweets with more than three stock tickers. 
Tweets with more than three stock tickers don't give good information about the specific stock in the query.
This method takes one single tweet as the `raw_tweet` parameter which is a full tweet status object and returns the `Tweet` object with the processed tweet text and the additional metadata. 

`save` method is a helper method to save tweets to a file. This method takes list of `Tweet` object and a file name and saves all the tweets text to the file. 

`Tweet` is a data class for holding each processed tweet. Each tweet is stored as an object with all the required metadata. 
This class defines the following fields 
* `text` - Processed tweet text 

* `likes_count` - Total number of likes for the tweet

* `retweets_count` - Total number of times the tweet was retweeted.

* `followers_count` - Total number of followers of the user who posted the tweet

* `sentiment_score` - Sentiment score of the tweet. This is computed by the ML model


### src/rank.py

This file contains the rank class that is used to select the most relevant Tweets from the entire collection retrieved by `src/twitter_client.py`, it also performs additional
processing.

When an instance of the class is created is needs the following parameters:

`corpus`: Set of tweets to be ranked

`query`: Query to be used to rank tweets in corpus

`output_file`: file to save ranked top_k tweets

`top_k`: number of ranked tweets to be returned and saved to output_file

`remove_stopwords`: Option to remove stopwords from set of tweets, default = True

`remove_digits`: Option to remove numbers from set of tweets, default = True

`remove_rare_words`: Option to remove word that appear only once, default = True

The method `pre_process_corpus` process the Tweets according to the options selected, it may remove stop words and digits from the documents.

The method `get_processed_corpus` returns the collection of Tweets ready to be ranked, it may remove rare words that appear only once in the entire collection.

Finally, `get_ranked_documents` collects the processed collection of `Tweets` and rank them using `Gensim` implementation of `BM25`. The process of creating the model to score the `Tweets` involves the creation of a `dictionary` of unique terms from the collection and the creation of a `BOW` representation of each `Tweet`; this is a list of tuples with the term id and frequency for each word, it uses the provided query to calculate the relevance of each `Tweet` and returns the `top_k` documents, it also saves them to a file.



### src/sentiment_analysis.py

This file handles everything related to sentiment classification. Specifically, it implements a recursive neural network using PyTorch to accomplish this. 

`TextDataset` class: This handles the tokenization of Tweets, and is also responsible for feeding the model input. A detailed description can be found in the comments. 

`RNN` (Recursive Neural Network) class: This class extends the default neural network and is responsible for building the architecture of the neural network, and handling the 'forward' phase of training (passing input through the various layers). Again, detailed descriptions can be found in the comments.

`preprocess` and `preprocess_string`: These functions are responsible for preprocessing training/test data, and raw tweets, respectively. Preprocessing is important for maximizing accuracy and ensuring all data is structured the same way.

`accuracy`: This function takes model output and ground truth labels and compares them to compute accuracy. This function is only used when a new model is created.

`train`: This contains the code that actually performs the training passes on the neural network. This function is only used when a new model is created.

`evaluate`: This compares the models predictions on test data to ground truth labels. This function is only used when a new model is created.

`predict`: This function is responsible for generating sentiment predictions on our tweets. It requires as input a pre-trained model, a vocabulary, and a list of Tweets. It preprocesses and loads the Tweets into the `TextDataset` class and feeds them through the model in order to obtain sentiment predictions.

`build_model`: This function is how `src/main.py` interfaces with this file to obtain a model. This function first loads the training dataset, as the vocabulary from this dataset is required regardless of whether a new model is to built or an old model is to be loaded. If an old model can be loaded, this function simply loads that model and returns it along with the vocabulary. If no old model can be loaded, it must build and train a new one. To do this, it then loads the test data and utilizes the above utility functions to perform the training and evaluation. This new model is then persisted and returned, along with the training vocabulary.

## Installation (Guide for Windows 10 ONLY)

It is recommended that you use Anaconda to run this project. This is primarily due to anacondas robust environment system. To install Anaconda, follow the instructions on the [Anaconda website](https://www.anaconda.com/products/individual). Once you have anaconda installed, open an Anaconda prompt and navigate to the directory where you have cloned this repository, and then navigate to the `src` directory. 

The next step is to setup an anaconda environment. To do this, run `conda create -n myenv python=3.9.7`. This will create a conda environment named `myenv` that has Python 3.9.7 installed. You can name this environment whatever you want, but for the purpose of this guide we will assume it was named `myenv`. Once you have done this, run `conda activate myenv` to enter the new environment.

In order to run the Python libraries that this project requires, you will need to have the developer tools for C++ 14 installed. For Windows machines, you can visit [Microsoft's website](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and click the `Download Build Tools` link to install Visual Studio, which will bring with it the necessary build tools. During the installation process, you will be prompted to select which tools to install. Be sure to at least select `Desktop Development with C++` under `Desktop & Mobile`. There are alternative ways to install these build tools, and they can be installed on non-Windows platforms, but these have not been validated.

The final step is to install the necessary Python packages. Note that this project utilizes PyTorch, which is greatly benefited by the presence of an Nvidia GPU. The process for setting up to use a GPU is incredibly involved, however, so this guide will simply install the CPU version of PyTorch. This project comes with a pre-built and trained neural net, so this should be fine, but if you for some reason need to re-build the model, you will want to figure out how to get the GPU version of PyTorch to work.

To install the necessary packages, run 
```
pip install tweepy
pip install gensim==3.8.3
pip install nltk
pip install torch
```

Alternatively, you could run `pip install -r ../requirements.txt`

If these installations go well, your environment should be prepared to run our project.

The final setup step is to unzip `Data/model_data.zip` so that the files `Data/test_data.csv` and `Data/training_data.csv` are present. These files are too large to upload to GitHub, so they had to be zipped first.

## Usage

To run our project, open an Anaconda prompt and activate the environment you created in the `Installation` section. Then, navigate to the `src` directory of our repository. It is important that you be in the `src` directory as we make use of relative paths. To run our project, simply run `python main.py`.

Our project makes use of the command line to interact with users, both for retrieving input and displaying out. When the program is first run, you will be prompted to enter the name of a company or a stock ticker. Once a valid stock ticker or company name has been entered, you will see an output that says `Outputs stored with run number X` where `X` is an integer. This indicates that the outputs of this execution are being stored in the `Tweets` directory with that number appended to the file names. 

During the course of execution, six files will be written. `all_tweets_X.txt` contains every Tweet that was retrieved from Twitter. `relevant_tweets_X.txt` contains only those Tweets which were found to be relevant to the entered stock. The top 25% most relevant Tweets are deemed to be relevant. `positive_tweets_X.txt`, `negative_tweets_X.txt`, and `neutral_tweets_X.txt` contain the relevant Tweets that were found to have positive, negative, and neutral sentiments, respectively. These files are sorted by degree of sentiment, with the most extreme at the start of the file. `score_X.txt` will contain the overall sentiment score.

After this line, you will see the lines 
```
Retrieving Tweets from Twitter...
Ranking Tweets...
Classifying Tweets...
Loading Training Data...
```

as the program progresses through its execution. At this point, two things could happen. If a model has been persisted, meaning `Data/model` exists, then that model will be loaded and the final output will be immediately displayed. If this file does not exist, a new model will be built. THIS IS NOT RECOMMENDED. Using CPU, this process will take several days. Using GPU, it will take several hours. Our project includes a model that we have already trained, and you should make use of it. 

Once a model has been either loaded or built, the final output will be displayed. This output will consist of the following: The line `Positive Tweets:` followed by a list of every relevant positive Tweet, sorted by decreasing positivity. The line `Negative Tweets:` followed by a list of every relevant negative Tweet, sorted by decreasing negativity. The line `Neutral Tweets:` followed by a list of every relevant neutral Tweet, sorted by decreasing negativity. And finally, the line `Sentiment score (ranges from -1 to 1): X` where `X` is the overall sentiment score extracted from the relevant Tweets. A sentiment score of 0 indicates a neutral overall sentiment, a positive sentiment score indicates a positive overall sentiment, and a negative sentiment score indicates a negative overall sentiment. The more extreme the sentiment score, the more extreme the overall sentiment. All of this output will also be written to files as described above.

## Example Use Case

An example use case for this project would be to determine the overall sentiment on Twitter about Apple's stock. This could be accomplished by running our project and entering either `aapl` or `apple` as input. This would have our project retrieve Tweets from Twitter, rank them by relevant to Apple's stock, sort the relevant Tweets by sentiment, and then compute an overall sentiment score for Apple's stock. We have included the output of a run of our project using `aapl` as input with the number 1 in the `Tweets` directory. This run was completed on 12/05/21.

## Group Member Contributions

### Zachary Oldham (zoldham2)
<ul>
    <li>Implemented sentiment_analysis.py which provides all the utilities for generating sentiment predictions over Tweets. Also located the training dataset used to train the sentiment model.</li>
    <li>Implemented main.py to connect all project components and interface with the user. Also located csv containing list of all stock tickers and company names.</li>
    <li>Wrote Installation and Usage sections of README.md. Also wrote README.md documentation for main.py and sentiment_analysis.py.</li>
</ul>

### Yogeswara Rao Lekkalapudi (yrl3)
* Implemented twitter_client.py which provides api for accessing twitter data.
* Preliminary preprocessing of tweets and providing a list of tweets with text and additional metadata. 
* Explored Tweepy and Twitter APIs to fetch the relevant content.
* README.md documentation for twitter_client.py and related documentation in the presentation.

### Luis Mariano Ovalle Castaneda (lo22)
<ul>
    <li>Implemented rank.py, which scores and selects Tweets relevant to our application from the entire retrieve collection. It also does additional Tweets processing like removing stop words using a list from the NLTK toolkit. </li>
    <li>Used Gensim toolkit for Tweets scoring on rank.py (BM25 function).</li>
    <li>README.md documentation for rank.py and presentation slides elaboration.</li>
</ul>

## Sources
<ul>
    <li>https://stockanalysis.com/stocks/ : Source of Data/stocks.csv, which contains a list of every stock symbol and company name.</li>
    <li>http://help.sentiment140.com/for-students : Source of Data/test_data.csv and Data/training_data.csv, which are used for training the neural network.</li>
</ul>