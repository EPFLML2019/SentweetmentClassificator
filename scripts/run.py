##################################################################
###   Script used to produce our best solution for tweet       ###
###                    sentiment analyis                       ###
### Authors: Arthur Passuello, Lucas Strauss, Francois Quellec ###
##################################################################


import numpy as np
import warnings

from preprocessing.embeddings import getGloveDict
from preprocessing.tools import load_csv_test_data, create_csv_submission
from preprocessing.tokenizer import tokenize
from models.sepCNN import sepCNN_Model

# Data input and output paths
POS_TRAIN_PATH = '../data/twitter-datasets/train_pos_full.txt' 
NEG_TRAIN_PATH = '../data/twitter-datasets/train_neg_full.txt' 
DATA_TEST_PATH = '../data/twitter-datasets/test_data.txt'
OUTPUT_PATH = 'predictions_out.csv'

# Remove Future Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def generateFinalPredictions():

	print("Loading Dataset: ", end="")
	pos_ids, pos_text_train = load_csv_test_data(POS_TRAIN_PATH)
	neg_ids, neg_text_train = load_csv_test_data(NEG_TRAIN_PATH)
	full_dataset = np.concatenate((pos_text_train, neg_text_train), axis=None)
	full_labels = np.concatenate((np.ones(len(pos_text_train)), -np.ones(len(pos_text_train))), axis=None)
	print("Done")

	print("Tokenizing Dataset: ", end="")
	all_tokens = [tokenize(tweet) for tweet in full_dataset]
	print("Done")

	print("Training Glove: ", end="")
	wv = getGloveDict(tokensList=all_tokens, size=200, window=10, min_count=10, workers=16, iters=25, trainedByStanford=False, bigram=False)
	print("Done")

	print("Building Model: ", end="")
	full_labels[full_labels <0] = 0
	model= sepCNN_Model(all_tokens, wv, tensorboard=False, useBigrams=False)
	print("Done")

	print("Training Model: ")
	model.train_model(all_tokens, full_labels, wv, batch_size=128, 
                epochs=7, dropout_rate=0.2, filters = 64, kernel_size= 7, pool_size=7, learning_rate=1e-3)
	print("Done")

	print("Load the dataset to predict: ", end="")
	test_ids, test_x = load_csv_test_data(DATA_TEST_PATH, has_ID=True)
	print("Done")

	print("Tokenize it: ", end="")
	test_tokens = [tokenize(tweet) for tweet in test_x]
	print("Done")

	print("Predict: ", end="")
	predictions = model.predict(test_tokens)
	print("Done")

	print("Save final preduction: ", end="")
	create_csv_submission(test_ids, predictions, OUTPUT_PATH)
	print("Done")
	
if __name__ == '__main__':
	generateFinalPredictions()
