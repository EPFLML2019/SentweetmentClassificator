import numpy as np
from nltk.tokenize import TweetTokenizer

# Load library
from nltk.corpus import stopwords
from gensim import *
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import gensim.downloader as api
from gensim.models import KeyedVectors
import re
import os.path
from gensim.models.phrases import Phrases, Phraser


# Top pos smiley used and not removed
pos_smiley = ["\(':", "\(':<", "\(';", "\(\*:", "\(\*;", "\(:", "\(;", "\(=", ":'\)", ":'\]", ":'\}", 
              ":\)", ":\*\)", ":\*:", ":-\]", ":-\}", ":\]", ":\}", ";'\)", ";'\]", ";\)", ";\*\)", ";-\}"
             , ";\]", ";\}", "=\)", "\(=", "<3", ":p", ":D", "xD", ":)"]


# Top neg smiley used and not removed
neg_smiley = ["\)':", "\)':<", "\)';", "\)=", "\)=<", "/':", "/';", "/-:", "/:", "/:<", "/;", 
              "/;<", "/=", ":'/", ":'@", ":'\[", ":'\\", ":'\{", ":'\|", ":\(", ":\*\(", ":\*\{", 
            ":\*\|", ":-/", ":-@", ":-\[", ":-\\", ":-\|", ":/", ":@", ":\[", ":\\", ":\{", ":\|"
             , ";'\(", ";'/", ";'\[", ";\*\{", ";-/", ";-\|", ";/", ";@", ";\[", ";\\", ";\{", ";\|"
             ,"=\(", "</3"]


# Top word without sentiment meaning nor negative form possible
stop_words= ["i", "you", "it", "she", "he", "we", "they", "a", "in", "to", "the", "and", "my", "me", "of", "for", "that", "this", "on", "so", "be", "just", "your", "at", "its", "im", ".", ",", ")", "'", "(", "or", "by", "am", "ve", "our", "\"", "<", ">", "&", "\\", ":", "-", ";", "/"]



def tokenizeTweet(tweet, stop_words=False, smiley_tag = False, strip_handles=True, reduce_len=True, preserve_case=False):
    tknzr = TweetTokenizer(strip_handles=strip_handles, reduce_len=reduce_len, preserve_case=preserve_case, ngram=1)

    # Tokenize
    tokens = tknzr.tokenize(tweet)
    
    if smiley_tag:
        for word in tokens:
            if word in pos_smiley:
                word = '<pos_smiley>'
            elif word in neg_smiley:
                word = '<neg_smiley>'
    
    if stop_words:
        tokens = [word for word in tokens if word not in stop_words and not word.isnumeric()]

    return tokens

def computeBigrams(tweetsTokenized):
    phrases = Phrases(tweetsTokenized)
    bigram = Phraser(phrases)

    return [bigram[tweet] for tweet in tweetsTokenized]


def getWord2VecDict(tokensList=None, size=200, window=10, min_count=2, workers=10, iters=10, train=False):
    filename = "KeyedW2V_size" + str(size) + "_window" + str(window) + "_min_count" + str(min_count) + "_workers" + str(workers) + "_iter" + str(iters) + ".kv"

    if os.path.isfile(filename):
        return KeyedVectors.load(filename, mmap='r')
    else:
        print("You haven't train Word2Vec already with those parameters, it can take some time...")
        if tokensList is None: 
            print("Please provide a tokensList in order to train the model")
        else:
            model = models.Word2Vec(
            tokensList,
            size=size,
            window=window,
            min_count=min_count,
            workers=workers,
            iter=iters)

            kw = model.wv
            kw.save(filename)
            del model
            return kw   
    return None


def getFasttextDict(tokensList=None, size=200, window=10, min_count=2, workers=10, iters=10):
    filename = "KeyedFT_size" + str(size) + "_window" + str(window) + "_min_count" + str(min_count) + "_workers" + str(workers) + "_iter" + str(iters) + ".kv"

    if os.path.isfile(filename):
        return KeyedVectors.load(filename, mmap='r')
    else:
        print("You haven't train Word2Vec already with those parameters, it can take some time...")
        if tokensList is None: 
            print("Please provide a tokensList in order to train the model")
        else:
            model = models.Word2Vec(
            tokensList,
            size=size,
            window=window,
            min_count=min_count,
            workers=workers,
            iter=iters)

            kw = model.wv
            kw.save(filename)
            del model
            return kw   
    return None

def getGloveDict(tokensList=None, size=200, window=10, min_count=2, workers=10, iters=10):
    pass

def generateTweetFeatures(word_dic, words):
        num_words = len(words)
        if num_words < 1:
            num_words = 1
            
        vector = np.zeros(word_dic.vector_size)
        for word in words:
            if word in word_dic.vocab:
                vector += word_dic[word]
        vector /= num_words
        return vector

def generateTweetsFeatures(tweetsTokenized, kv):
    return np.array([generateTweetFeatures(kv, words) for words in tweetsTokenized])


