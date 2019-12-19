##################################################################
###     Implementation of a Bidirectional LSTM to predict      ###
###          negative or positive sentiment on tweets          ###
### Authors: Arthur Passuello, Lucas Strauss, Francois Quellec ###
##################################################################


import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Embedding, Dense, Dropout, Activation, GRU, LSTM, Bidirectional, Flatten, GlobalMaxPool1D
import tensorflow as tf
from preprocessing.tools import *
from preprocessing.bigrams import bigramGenerator

class BLSTM_Model:
    def __init__(self, tweetsTokenized, embedding_vectors, use_gru = False, tensorboard=False, useBigrams=False):
        self.embedding_vectors = embedding_vectors
        self.useBigrams = useBigrams

        # Generate Bigrams if needed
        if self.useBigrams:
            self.bigram = bigramGenerator(tweetsTokenized)
            tweetsTokenized = [self.bigram[tweet] for tweet in tweetsTokenized]

        # Remove unknown words
        tweetsTokenized = [list(filter(lambda i: i in embedding_vectors, tweet)) for tweet in tweetsTokenized]


        # Fit a tokenizer on the corpus to transform each unique word in unique int identifier and feed the model with it
        self.tokenizer_obj = Tokenizer()
        self.tokenizer_obj.fit_on_texts(tweetsTokenized)

        # Compute the maximum number of words in test tweets
        self.max_length = max([len(tweet_tokens) for tweet_tokens in tweetsTokenized])
        self.model = None
        
        self.use_gru = use_gru
        self.tensorboard = tensorboard

        

    def train_model(self, tweetsTokenized, labels, embedding_vectors, batch_size=128, epochs=5, dropout_embedding=0.4, dropout_relu=0.5):
        
        # Generate Bigrams if needed
        if self.useBigrams:
            tweetsTokenized = [self.bigram[tweet] for tweet in tweetsTokenized]
            self.max_length = max([len(tweet_tokens) for tweet_tokens in tweetsTokenized])

        tweetsTokenized = [list(filter(lambda i: i in embedding_vectors, tweet)) for tweet in tweetsTokenized]

        # Transform each unique word in unique int identifier
        sequences = self.tokenizer_obj.texts_to_sequences(tweetsTokenized)

        # Pad the tweet to have all the same size
        tweet_padded = pad_sequences(sequences, maxlen=self.max_length)

        # Construct our model with keras
        self.model = Sequential()

        # Add the embedding layer with our trained embedding matrix
        embedding_layer = Embedding(input_dim=embedding_vectors.syn0.shape[0] , output_dim=embedding_vectors.syn0.shape[1], weights=[embedding_vectors.syn0], 
                                input_length=tweet_padded.shape[1])
        self.model.add(embedding_layer)

        # Add dropout to prevent overfitting
        self.model.add(Dropout(dropout_embedding))

        # Add LSTM or GRU layer
        if self.use_gru:
            self.model.add(GRU(128))
        else:
            self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
            
        #self.model.add(GlobalMaxPool1D()) #Or at the same place as Flatten()
        self.model.add(Dense(32))
        self.model.add(Dropout(dropout_relu))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
        
        callbacks = [reduce_lr]
        if self.tensorboard:
            if self.use_gru:
                logdir = 'logs/gru_dropout_embedding=' + str(dropout_embedding) + "_dropout_relu="+dropout_relu+"_batch_size="+batch_size+"_epochs="+str(epochs)
            else:
                logdir = 'logs/blstm_dropout_embedding=' + str(dropout_embedding) + "_dropout_relu="+str(dropout_relu)+"_batch_size="+str(batch_size)+"_epochs="+str(epochs)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
            callbacks.append(tensorboard_callback)
        
        print (self.model.summary())

        # Train the model
        self.model.fit(tweet_padded, labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True, callbacks=callbacks)

    def predict(self, tweetsTokenized):
        if self.useBigrams:
            tweetsTokenized = [self.bigram[tweet] for tweet in tweetsTokenized]         

        # Transform each unique word in unique int identifier
        sequences = self.tokenizer_obj.texts_to_sequences(tweetsTokenized)

        # Pad the tweet to have all the same size
        tweet_pad = pad_sequences(sequences, maxlen=self.max_length)

        # Predict
        predictions = self.model.predict(x=tweet_pad)

        return predict_labels(predictions, treshold = 0.5)

    def save(self, filename):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(filename + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(filename + ".h5")
        print("Saved model to disk")

        

