# Twitter Stock Sentiment Analysis

## Overview

## Installation (Guide for Windows 10 ONLY)

It is reccommended that you use Anaconda to run this project. This is primarily due to anacondas robust environment system. To install Anaconda, follow the instructions on the [Anaconda website](https://www.anaconda.com/products/individual). Once you have anaconda installed, open an Anaconda prompt and navigate to the directory where you have cloned this repository, and then navigate to the `src` directory. 

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

as the program progesses through its execution. At this point, two things could happen. If a model has been persisted, meaning `Data/model` exists, then that model will be loaded and the final output will be immediately displayed. If this file does not exist, a new model will be built. THIS IS NOT RECCOMMENDED. Using CPU, this process will take several days. Using GPU, it will take several hours. Our project includes a model that we have already trained, and you should make use of it. 

Once a model has been either loaded or built, the final output will be displayed. This output will consist of the following: The line `Positive Tweets:` followed by a list of every relevant positive Tweet, sorted by decreasing positivity. The line `Negative Tweets:` followed by a list of every relevant negative Tweet, sorted by decreasing negativity. The line `Neutral Tweets:` followed by a list of every relevant neutral Tweet, sorted by decreasing negativity. And finally, the line `Sentiment score (ranges from -1 to 1): X` where `X` is the overall sentiment score extracted from the relevant Tweets. A sentiment score of 0 indicates a neutral overall sentiment, a positive sentiment score indicates a positive overall sentiment, and a negative sentiment score indicates a negative overall sentiment. The more extreme the sentiment score, the more extreme the overall sentiment. All of this output will also be written to files as described above.

## Example Use Case

An example use case for this project would be to determine the overall sentiment on Twitter about Apple's stock. This could be accomplished by running our project and entering either `aapl` or `apple` as input. This would have our project retrieve Tweets from Twitter, rank them by relevant to Apple's stock, sort the relevant Tweets by sentiment, and then compute an overall sentiment score for Apple's stock. We have included the output of a run of our project using `aapl` as input with the number 1 in the `Tweets` directory. This run was completed on 12/05/21.