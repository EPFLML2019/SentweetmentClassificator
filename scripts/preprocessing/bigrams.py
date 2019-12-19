##################################################################
###       Generation of Bigrams to feed a Neural Network       ###
###                                                            ###
### Authors: Arthur Passuello, Lucas Strauss, Francois Quellec ###
##################################################################

from gensim.models.phrases import Phrases, Phraser
import numpy as np
import csv
from preprocessing.tools import load_csv_test_data
from preprocessing.tokenizer import tokenize
import pandas as pd
import sys
import ntpath



def bigramGenerator(tweets_collection_tokenized):
    """ Function that return a bigram generator trained on a collection of tweet tokenized """ 

    phrases = Phrases(tweets_collection_tokenized)
    bigram = Phraser(phrases)
    return bigram

def computeBigrams(tweetsTokenized):
    """ Function that transform a list of phrases tokenized by a list of phrase tokenized with in bigrams """ 
    
    phrases = Phrases(tweetsTokenized)
    bigram = Phraser(phrases)

    return [bigram[tweet] for tweet in tweetsTokenized]


if __name__ == '__main__':
    _, filename = sys.argv
    tweets = load_csv_test_data(filename)[1]
    # Assume that tokens are delimited by spaces, may want to use our tokenizer before
    tokens = [tweet.split(" ") for tweet in tweets]

    # Generate tokens
    bigrams_tokens = computeBigrams(tokens)

    # Write to file
    with open('bigram_' + ntpath.basename(filename), 'a+') as output:
        for bigram_tweet in bigrams_tokens:
            output.write(" ".join(bigram_tweet) + "\n")
        print ("DONE!")
 
