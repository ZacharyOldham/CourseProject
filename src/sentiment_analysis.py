from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import random
import string
import os
import re

# Important constants for building the model and parsing data
ALL_PUNCT = string.punctuation
PAD = '<PAD>'
END = '<END>'
UNK = '<UNK>'
NUM = '<NUM>'
NAME = '<NAME>'
URL = '<URL>'
STOCK = '<STOCK>'
THRESHOLD = 5
MAX_LEN = 100
BATCH_SIZE = 256
LEARNING_RATE = 5e-4
EPOCHS = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This dataset class performs all of the tokenizing and word mapping necessary to feed data into the neural network
class TextDataset(data.Dataset):

    # Initialize the dataset
    # examples: A list of labeled example tweets
    # split: A string representing the mode this dataset will be used for
    # threshold: The cutoff for converting rare words into UNK tokens
    # max_len: The max length of any example
    # idx2word: A mapping from indicies to words. Not necessary if split="train"
    # word2idx: A mapping from words to indicies. Not necessary if split="train"
    def __init__(self, examples, split, threshold, max_len, idx2word=None, word2idx=None):

        # Save data
        self.examples = examples
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.threshold = threshold
        self.max_len = max_len

        # Dictionaries
        self.idx2word = idx2word
        self.word2idx = word2idx
        if split == 'train':
            self.build_dictionary()
        self.vocab_size = len(self.word2idx)
        
        # Convert text to indices
        self.textual_ids = []
        self.convert_text()
    
    # Build the dictionaries that map from words to indicies and vice versa
    def build_dictionary(self): 

        # If the dataset is not in training mode, this should not be done
        assert self.split == 'train'
        
        # Initialiez the dictionaries
        self.idx2word = {0:PAD, 1:END, 2: UNK}
        self.word2idx = {PAD:0, END:1, UNK: 2}

        # Gather token frequencies
        freq = {}
        for label, doc in self.examples:
            for word in doc:
                # word = word.lower()
                if word in freq:
                    freq[word] += 1
                else:
                    freq[word] = 1

        # Add tokens to the dictionaries only if they occur more than THRESHOLD
        cur_idx = 3
        for word in freq:
            if freq[word] >= self.threshold:
                self.idx2word[cur_idx] = word
                self.word2idx[word] = cur_idx
                cur_idx += 1
    
    # For each document, create a mapping of each token to its corresponding ID
    def convert_text(self):

        # Iterate over documents
        for label, doc in self.examples:
            converted_doc = []

            # Build ID list for this document
            for word in doc:
                word = word.lower()
                if word in self.word2idx:
                    converted_doc.append(self.word2idx[word])
                else:
                    converted_doc.append(self.word2idx[UNK])
            converted_doc.append(self.word2idx[END])
            self.textual_ids.append(converted_doc)

    # obtain a list of ids representing a document by the index of the document
    # idx: the index of the document
    def get_text(self, idx):

        # Get the pre-computed mapping
        if idx > len(self.textual_ids):
            print(len(self.examples))
            print(len(self.textual_ids))
            print("INDEX OUT OF BOUNDS")
        review = self.textual_ids[idx]

        # Add padding as necessary and return
        if len(review) < self.max_len:
            review.extend([self.word2idx[PAD]] * (self.max_len - len(review)))
        return torch.LongTensor(review[:self.max_len])
    
    # get the ground truth label for a document by the index of the document
    # idx: the index of the document
    def get_label(self, idx):
        label, doc = self.examples[idx]
        if label == 1:
            return torch.tensor(1)
        else:
            return torch.tensor(0)

    # overload of the len() function. returns the number of examples.
    def __len__(self):
        return len(self.examples)
    
    # overload to allow for iteration over (text, label) pairs
    def __getitem__(self, idx):
        return self.get_text(idx), self.get_label(idx)

# Neural network class that uses a bidirectional recurrent neural network to perform classification on a string
class RNN(nn.Module):
    # Initialize the neural net
    # vocab_size: the number of unique tokens
    # embed_size: the number of nodes used to create the embedding
    # hidden_size: the size of the hidden layer
    # num_layers: the number of layers in the rnn
    # bidirection: whether or not the rnn is bidirectional
    # dropout: the rate at which to apply dropout
    # num_classes: the number of classes the strings should be classified into
    # pad_idx: the index of the padding in the vocab
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, dropout, num_classes, pad_idx):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Create embed layer
        self.embedding_layer = torch.nn.Embedding(vocab_size, embed_size, pad_idx)

        # Create a rnn
        self.gru_layer = torch.nn.GRU(bidirectional=bidirectional, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, input_size=embed_size)
        
        # Create a dropout
        self.dropout_layer = torch.nn.Dropout(dropout)

        # Define a linear layer 
        if self.bidirectional:
            self.linear_layer = torch.nn.Linear(2*hidden_size, num_classes)
        else:
            self.linear_layer = torch.nn.Linear(hidden_size, num_classes)
        self.prev_hidden_state = None

    # The forward step of the nn, where values are propagated through the neural network
    # texts: the texts that are being passed through the net on this step
    def forward(self, texts):

        # Pass texts through the embedding layter
        #   Resulting: shape: [batch_size, max_len, embed_size]
        embedding_output = self.embedding_layer(texts)
        

        # Pass the embedding through the rnn
        #   See PyTorch documentation for resulting shape for nn.GRU
        hidden_state = None
        _, hidden_state = self.gru_layer(embedding_output)
        
        # Concatenate the outputs of the last timestep for each direction
        #   Resulting shape: [batch_size, num_dirs*hidden_size]
        gru_output = None
        if self.bidirectional:
            gru_output = torch.cat([hidden_state[-2], hidden_state[-1]], 1)
        else:
            gru_output = hidden_state[-1]
        
        # Apply dropout
        dropout_output = self.dropout_layer(gru_output)

        # Pass the current state throgh the linear layer to get final output
        #   Resulting shape: [batch_size, num_classes]
        final_output = self.linear_layer(dropout_output)
        return final_output

# preprocess a line from the training or test data files
# line: a line from one of these files
def preprocess(line):

    # Split the line into its parts
    line = line.strip()
    row = line.split(",", 5)
    result = []

    # Update the label to be either 0 or 1
    result.append(int(row[0][1]))
    if result[0] == 2:
        return None
    if result[0] == 4:
        result[0] = 1

    # Preprocess the text of the tweet
    result.append(preprocess_string(row[5][1:-1]))
    if result[1] is None:
        return None
    else:
        return result

# perform preprocessing on the raw text of a tweet
# in_str: the raw text of the tweet
def preprocess_string(in_str):
    out_str = in_str.lower()

    # Remove retweet info
    out_str = re.sub(r'^rt @\w+:? ?', '', out_str)

    # Replace URLs with uniform tokens
    out_str = re.sub(r'http\S+', URL, out_str)

    # Replace stock tickers with uniform tokens
    out_str = re.sub(r'\$[a-z]+', STOCK, out_str)

    # Remove punctuation
    for punct in ALL_PUNCT:
        if punct != "@":
            out_str = out_str.replace(punct, "")
    out_str = out_str.replace(URL[1:-1], URL)
    out_str = out_str.replace(STOCK[1:-1], STOCK)
    out_toks = out_str.split()

    # Replace numbers and names with uniform tokens
    for i in range(len(out_toks)-1, -1, -1):
        if out_toks[i].isnumeric():
            out_toks[i] = NUM
        elif out_toks[i][0] == "@":
            out_toks[i] = NAME
        else:
            out_toks[i] = out_toks[i].replace("@", "")
            if len(out_toks[i]) == 0:
                out_toks.pop(i)

    # Only return if length is less than MAX_LEN
    if len(out_toks) >= MAX_LEN:
        return None
    else:
        return out_toks

# Compute the accuracy of a classification by comparing the output of the model with the ground truth labels
# output: the output of the classifier
# labels: the groud truth labels
def accuracy(output, labels):
    preds = output.argmax(dim=1)
    correct = (preds == labels).sum().float()
    acc = correct / len(labels)
    return acc

# Train the model for the given number of epochs using data in the loader and the given optimizer and criterion to determine the optimal values
# model: the model to train
# loader: a dataloader that contain all of the training data
# optimizer: the optimizer to use while training
# criterion: the criterion to use while training
def train(model, epochs, loader, optimizer, criterion):
    model.train()

    # Perform one pass per epoch
    for cur_epoch in range(0, epochs):
        epoch_loss = 0
        epoch_acc = 0

        # Iterate over each training item
        for texts, labels in loader:
            texts = texts.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Obtain model predictions
            output = model(texts)
            acc = accuracy(output, labels)
            loss = criterion(output, labels)

            # Update the model
            loss.backward()
            optimizer.step()

            # Store the loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
        # Output updated statistics after this most recent training pass
        print("Epoch " + str(cur_epoch + 1) + " Loss: " + str(epoch_loss / len(loader)))
        print("Epoch " + str(cur_epoch + 1) + " Acc: " + str(100 * epoch_acc / len(loader)))
    print("Done Training")

# Evaluate the performance of the model
# model: the model to evaluate
# loader: a dataloader containing the data to use to do the evaluation
# criterion: the criterion on which to evaluate the model
def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []

    # Iterate over each test item
    for texts, labels in loader:
        texts = texts.to(device)
        labels = labels.to(device)

        # Obtain model predictions
        output = model(texts)
        acc = accuracy(output, labels)
        pred = output.argmax(dim=1)
        all_predictions.append(pred)
        loss = criterion(output, labels)

        # Update statistics
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    # Output statistics
    acc = 100*epoch_acc/len(loader)
    loss = epoch_loss/len(loader)
    print("Test Loss: " + str(loss))
    print("Test Acc: " + str(acc))
    predictions = torch.cat(all_predictions)

    # Return results
    return predictions, acc, loss

# Generate predictions for the given tweets using the given model and vocab
# model: the model to use to generate predictions
# idx2word: the mapping of indexes to words to use for the vocab
# word2idx: the mapping of words to indexes to use for the vocab
# tweets: the list of tweets to generate predictions for
def predict(model, idx2word, word2idx, tweets):
    data = [[0, preprocess_string(x)] for x in tweets]

    # Put the data in to a data loader
    dataset = TextDataset(data, "test", THRESHOLD, MAX_LEN, idx2word, word2idx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    model.eval()

    # Gather predictions
    all_predictions = []
    for tweet_batch, _ in loader:
        tweet_batch = tweet_batch.to(device)
        out = model(tweet_batch)
        preds = []
        for i in range(0, len(out)):
            preds.append(out[i][1] - out[i][0])
            preds[-1] = preds[-1].item()
        all_predictions.extend(preds)
    return all_predictions

# Build a model using the training data
# force_rebuild: if false, the model will be loaded from disc if possible. If true, the model will be rebuild even if an existing model has been persisted
def build_model(force_rebuild=False):

    # Load training data
    print("Loading Training Data...")
    training_data = []
    with open("../Data/training_data.csv", "r") as f:
        for line in f:
            row = preprocess(line)
            if row is not None:
                training_data.append(row)
    training_dataset = TextDataset(training_data, "train", THRESHOLD, MAX_LEN)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    # Check if we need to build a new model or load one. If we can load, do that
    if not force_rebuild and os.path.exists("../Data/model"):
        model = torch.load("../Data/model", map_location=device)
        return (model, training_dataset.idx2word, training_dataset.word2idx)

    # Load test data
    print("Loading Test Data...")
    test_data = []
    with open("../Data/test_data.csv", "r") as f:
        for line in f:
            row = preprocess(line)
            if row is not None:
                test_data.append(row)
    test_dataset = TextDataset(test_data, "test", THRESHOLD, MAX_LEN, training_dataset.idx2word, training_dataset.word2idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    print("Building model using " + ("GPU" if torch.cuda.is_available() else "CPU"))

    # Build RNN
    print("Building RNN...")
    model =   RNN(vocab_size=training_dataset.vocab_size,
            embed_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.5,
            num_classes=2,
            pad_idx=training_dataset.word2idx[PAD])
    model = model.to(device)

    # Train the rnn
    print("Training RNN...")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train(model, EPOCHS, training_loader, optimizer, criterion)
    print("Done training RNN")
    evaluate(model, test_loader, criterion)

    # Persist and return the new model
    torch.save(model, "../Data/model")
    return (model, training_dataset.idx2word, training_dataset.word2idx)