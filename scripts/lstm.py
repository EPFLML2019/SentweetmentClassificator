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
from tools import *

class LSTM_Model:
    def __init__(self, tweetsTokenized, use_gru = False, tensorboard=False):
        self.tokenizer_obj = Tokenizer()
        self.tokenizer_obj.fit_on_texts(tweetsTokenized)

        # Compute the maximum number of words in test tweets
        self.max_length = max([len(tweet_tokens) for tweet_tokens in tweetsTokenized])
        self.model = None
        
        self.use_gru = use_gru
        self.tensorboard = tensorboard

    def train_model(self, tweetsTokenized, labels, embedding_vectors, batch_size=128, epochs=5):
        # Transform each unique word in unique int identifier
        sequences = self.tokenizer_obj.texts_to_sequences(tweetsTokenized)

        # Pad the tweet to have all the same size
        tweet_padded = pad_sequences(sequences, maxlen=self.max_length)

        # Construct our model with keras
        self.model = Sequential()

        # Add the embedding layer with our trained embedding matrix
        embedding_layer = Embedding(input_dim=embedding_vectors.syn0.shape[0], output_dim=embedding_vectors.syn0.shape[1], weights=[embedding_vectors.syn0], 
                                input_length=tweet_padded.shape[1])
        self.model.add(embedding_layer)

        # Add dropout to prevent overfitting
        self.model.add(Dropout(0.4))

        # Add LSTM or GRU layer
        if self.use_gru:
            self.model.add(GRU(128))
        else:
            self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
            
        #self.model.add(GlobalMaxPool1D()) #Or at the same place as Flatten()
        self.model.add(Dense(32))
        self.model.add(Dropout(0.5))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
        
        callbacks = [reduce_lr]
        if self.tensorboard:
            if self.use_gru:
                logdir = 'logs/gru'
            else:
                logdir = 'logs/blstm'
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
            callbacks.append(tensorboard_callback)
        
        print (self.model.summary())

        # Train the model
        self.model.fit(tweet_padded, labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True, callbacks=callbacks)

    def predict(self, tweetsTokenized):
        sequences = self.tokenizer_obj.texts_to_sequences(tweetsTokenized)
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

        

