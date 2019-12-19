##################################################################
###       Script to run bert algorithm on tweets               ###
###                                                            ###
### Authors: Arthur Passuello, Lucas Strauss, Francois Quellec ###
##################################################################

# Load library
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from nltk.corpus import stopwords
from gensim import *
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import gensim.downloader as api
import re
import ktrain
from ktrain import text
from tools import *

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

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(full_dataset, full_labels, test_size=0.1, train_size=0.1)
print("Training embedding:")
(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=x_train, y_train=y_train,
                                                                       x_test=x_test, y_test=y_test,
                                                                       class_names=[-1, 1],
                                                                       preprocess_mode='bert',
                                                                       maxlen=350, 
                                                                       max_features=35000)

print("Create model:")
model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)
learner = ktrain.get_learner(model, train_data=(x_train, y_train), batch_size=6)
print("Training model:")
learner.fit_onecycle(2e-5, 1)