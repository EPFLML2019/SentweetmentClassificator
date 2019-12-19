##################################################################
###   Train or download of different embeddings solutions      ###
###                                                            ###
### Authors: Arthur Passuello, Lucas Strauss, Francois Quellec ###
##################################################################

import numpy as np
from nltk.tokenize import TweetTokenizer
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
from gensim.test.utils import datapath
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os

GLOVE_PATH = "GloVe-1.2/build/"

def getWord2VecDict(tokensList=None, size=200, window=10, min_count=2, workers=10, iters=10):
    """ Return a dictionnary of embeddings for the word in tokensList using word2vec algorithm """
    filename = "KeyedW2V_size" + str(size) + "_window" + str(window) + "_min_count" + str(min_count) + "_workers" + str(workers) + "_iter" + str(iters) + '.kv'

    if os.path.isfile(filename):
        return KeyedVectors.load(filename, mmap='r')
    else:
        print("You haven't train Word2Vec already with those parameters, it can take some time...")
        if tokensList is None: 
            print("Please provide a tokensList in order to train the model")
        else:
            model = models.Word2Vec(tokensList, size=size, window=window, min_count=min_count, workers=workers, iter=iters)

            kw = model.wv
            kw.save(filename)
            del model
            return kw   
    return None


def getFasttextDict(tokensList=None, size=200, window=10, min_count=2, workers=10, iters=10):
    """ Return a dictionnary of embeddings for the word in tokensList using FastText algorithm """

    filename = "KeyedFT_size" + str(size) + "_window" + str(window) + "_min_count" + str(min_count) + "_workers" + str(workers) + "_iter" + str(iters) + '.kv'

    if os.path.isfile(filename):
        return KeyedVectors.load(filename, mmap='r')
    else:
        print("You haven't train FastText already with those parameters, it can take some time...")
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

def getGloveDict(tokensList=None, size=100, window=10, min_count=10, workers=16, iters=25, trainedByStanford=False, bigram=False):
    """ Return a dictionnary of embeddings for the word in tokensList using Glove algorithm or a pre-train one on tweets form Stanford University"""

    if trainedByStanford:
        filename = "glove-twitter-" + str(DIM)
        if os.path.isfile(filename):
            return KeyedVectors.load(filename, mmap='r')

        wv = api.load(filename).wv
        wv.save(filename)
        return wv

    filename = "KeyedGlove" +("_bigram" if bigram else "") + "_size" + str(size) + "_window" + str(window) + "_min_count" + str(min_count) + "_workers" + str(workers) + "_iter" + str(iters) + ".kv"

    if os.path.isfile(filename):
        return KeyedVectors.load(filename, mmap='r')
    else:
        print("You haven't train Word2Vec already with those parameters, it can take some time...")
        if tokensList is None: 
            print("Please provide a tokensList in order to train the model")
        else:
            vectors_filename = "vectors" +("_bigram" if bigram else "") +"_dim-" + str(size)

            success = trainGlove(tokensList,
                        vectors_filename,
                        size=size,
                        window=window,
                        min_count=min_count,
                        workers=workers,
                        iter=iters,
                        bigram=bigram)

            if not success:
                return None

            tmp_file = get_tmpfile("temp_glove.txt")

            _ = glove2word2vec(vectors_filename + ".txt", tmp_file)

            wv = KeyedVectors.load_word2vec_format(tmp_file)

            os.remove(vectors_filename + ".txt")

            # Normalize 
            wv.init_sims(replace=True)

            wv.save(filename)
            return wv   

    return None

def generateTweetFeatures(word_dic, words):
    """ Generate features for a tweet by averaging their word embedding """
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
    """ Generate features for a collection of tweet by averaging their word embedding """
    return np.array([generateTweetFeatures(kv, words) for words in tweetsTokenized])


def trainGlove(tokensList, vectors_filename, size,window,
                min_count,workers,iter,bigram): 
    """ Train Stanford Glove algorithm  """

    with open("temp_tweet_tokens_spaces.txt", "w+") as fp:
        for tweet in tokensList:
            fp.write(" ".join(tweet) + " \n")

    if os.path.isdir(GLOVE_PATH):
        os.system("./" + GLOVE_PATH + "vocab_count -verbose 2 -max-vocab 100000 -min-count "+str(min_count)+" < temp_tweet_tokens_spaces.txt > vocab.txt")
        os.system("./" + GLOVE_PATH + "cooccur -verbose 2 -symmetric 0 -window-size "+str(window)+" -vocab-file vocab.txt -memory 8.0 -overflow-file tempoverflow < temp_tweet_tokens_spaces.txt > cooccurrences.bin")
        os.system("./" + GLOVE_PATH + "shuffle -verbose 2 -memory 8.0 < cooccurrences.bin > cooccurrence.shuf.bin")
        os.system("./" + GLOVE_PATH + "./glove -input-file cooccurrence.shuf.bin -vocab-file vocab.txt -save-file " + vectors_filename + " -verbose 2 -vector-size "+str(size)+" -threads "+str(workers)+" -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 0 -model 2")

        os.remove("vocab.txt")
        os.remove("temp_tweet_tokens_spaces.txt")
        os.remove("cooccurrences.bin")
        os.remove("cooccurrence.shuf.bin")

        return True
    else:
        print("Directory " + GLOVE_PATH + " not found, please download a the root of the project the glove implementation from https://nlp.stanford.edu/projects/glove/")
        return False

    
