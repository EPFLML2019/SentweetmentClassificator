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

def getWord2VecDict(tokensList=None, size=200, window=10, min_count=2, workers=10, iters=10, train=True):
    filename = "KeyedW2V_size" + str(size) + "_window" + str(window) + "_min_count" + str(min_count) + "_workers" + str(workers) + "_iter" + str(iters) + ".kv"

    if os.path.isfile(filename) and not train:
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


def getFasttextDict(tokensList=None, size=200, window=10, min_count=2, workers=10, iters=10, train=False):
    filename = "KeyedFT_size" + str(size) + "_window" + str(window) + "_min_count" + str(min_count) + "_workers" + str(workers) + "_iter" + str(iters) + ".kv"

    if os.path.isfile(filename) and not train:
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


