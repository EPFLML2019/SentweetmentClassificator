test_size=0.1
train_size=0.1
DIM = 50
BIGRAM = False


from preprocessing.embeddings import *
from preprocessing.tools import *
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
import os.path
from preprocessing.tokenizer import *

# Load library
from nltk.corpus import stopwords
from gensim import *
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import gensim.downloader as api
import re
from sklearn.metrics import accuracy_score

# Data input and output paths
POS_TRAIN_PATH = '../data/twitter-datasets/train_pos_full.txt' 
NEG_TRAIN_PATH = '../data/twitter-datasets/train_neg_full.txt' 
DATA_TEST_PATH = '../data/twitter-datasets/test_data.txt'
OUTPUT_PATH = 'predictions_out.csv'

TOKENS_PATH = "../saved_gen_files/all_tokens.txt"
FULL_TRAIN_TWEET_VECTORS = "../saved_gen_files/train_tweet_vectors.txt"


pos_ids, pos_text_train = load_csv_test_data(POS_TRAIN_PATH)
neg_ids, neg_text_train = load_csv_test_data(NEG_TRAIN_PATH)
full_dataset = np.concatenate((pos_text_train, neg_text_train), axis=None)
full_labels = np.concatenate((np.ones(len(pos_text_train)), -np.ones(len(pos_text_train))), axis=None)


with open(TOKENS_PATH, 'rb') as f:
        all_tokens = pickle.load(f)


# Generate bigrams
#all_tokens = computeBigrams(all_tokens)

from gensim.test.utils import datapath
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
if BIGRAM:
	glove_file = '../data/self_trained_gloves/vectors_bigram_d'+str(DIM)+'.txt'
else:	
	glove_file = '../data/self_trained_gloves/vectors_d'+str(DIM)+'.txt'
tmp_file = get_tmpfile("test_word2vec.txt")

_ = glove2word2vec(glove_file, tmp_file)

wv = KeyedVectors.load_word2vec_format(tmp_file)

# Normalize 
wv.init_sims(replace=True)

labels = full_labels
labels[labels<0] = 0

X_train, X_test, y_train, y_test = train_test_split(all_tokens, labels, test_size=test_size, train_size=train_size, random_state=1)


use_tensorboard = False

#######################################
####		     sep-CNN		   ####
#######################################
from models.sepCNN import *
print('Training with sepCNN')
model= sepCNN_Model(all_tokens, wv, tensorboard=True, useBigrams=BIGRAM)
model.train_model(X_train, y_train, wv, batch_size=128, epochs=5)
model.save("sepCnn_bigram")
# Test the model
predictions = model.predict(X_test)
predictions[predictions<0] = 0
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
model.save("sepCnn_bigram")

#######################################
####		     BLSTM 			   ####
#######################################


from models.lstm import *
print('Training with BLSTM')
model= LSTM_Model(all_tokens, wv, use_gru=False, tensorboard=False, useBigrams=BIGRAM)
model.train_model(X_train, y_train, wv, batch_size=128, epochs=5)
model.save("blstm_bigram")
# Test the model
predictions = model.predict(X_test)
predictions[predictions<0] = 0
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
model.save("blstm_bigram")

#######################################
####		     SVM 			   ####
#######################################

# Convert tweet in features with previous embedding system
print('Training with SVM')
X_train, X_test, y_train, y_test = train_test_split(all_tokens, full_labels, test_size=test_size, train_size=train_size, random_state=1)
X_train_svm = generateTweetsFeatures(X_train, wv)
X_test_svm = generateTweetsFeatures(X_test, wv)


from sklearn import svm

clf_svm = svm.SVC(gamma='scale')
clf_svm.fit(X_train_svm, y_train)
predict_svm = clf_svm.predict(X_test_svm)
predict_svm = predict_labels(predict_svm)
predict_svm[predict_svm<0] = 0
y_test[y_test<0] = 0
print(classification_report(y_test, predict_svm))
print(accuracy_score(y_test, predict_svm))

from joblib import dump, load
dump(clf_svm, 'svm_bigram.joblib') 


