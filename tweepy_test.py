import tweepy.asynchronous
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities

auth = tweepy.AppAuthHandler("xlinnUvFlgoU2JQlynXU5Vx14", "36Pdrs0qNMqYUg8DBbVBRAiSC4i5yNb27Xo6Tm9XGg5JlNpBeT")
api = tweepy.API(auth)

documents = []

f = open("tweets.txt", "w")

# Search for tweets and save the in tweets.txt

for tweet in tweepy.Cursor(api.search_tweets, q="$NVDA stock", lang="en", tweet_mode='extended').items(500):
    f.write(str(tweet.full_text).replace("\n", " "))
    documents.append(str(tweet.full_text).replace("\n", " "))
    f.write("\n")

f.close()

stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print(dictionary)

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

doc = "NVDA stock market Nvidia performance"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]  # convert the query to LSI space
print(vec_lsi)

index = similarities.MatrixSimilarity(lsi[corpus])

sims = index[vec_lsi]  # perform a similarity query against the corpus
print(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sort tweets using similarity
for doc_position, doc_score in sims:
    print(doc_score, documents[doc_position])
