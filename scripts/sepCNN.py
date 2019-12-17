from tools import *

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, Dropout, Activation, GRU, LSTM, Bidirectional, Flatten, GlobalMaxPool1D


class sepCNN_Model:
  """docstring for sepCNN_Model"""
  def __init__(self, tweetsTokenized, tensorboard=False):
    self.tokenizer_obj = Tokenizer()
    self.tokenizer_obj.fit_on_texts(tweetsTokenized)

    # Compute the maximum number of words in test tweets
    self.max_length = max([len(tweet_tokens) for tweet_tokens in tweetsTokenized])
    self.model = None
    self.tensorboard = tensorboard

  def train_model(self, tweetsTokenized, labels, embedding_vectors, batch_size=128, 
                epochs=5, dropout_rate=0.2, filters = 64, kernel_size= 7, pool_size=7, learning_rate=1e-3):
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
    self.model.add(Dropout(rate=dropout_rate))
    self.model.add(SeparableConv1D(filters=filters,
                                      kernel_size=kernel_size,
                                      activation='relu',
                                      bias_initializer='random_uniform',
                                      depthwise_initializer='random_uniform',
                                      kernel_initializer='glorot_uniform',
                                      padding='same'))
    self.model.add(SeparableConv1D(filters=filters,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              kernel_initializer='glorot_uniform',
                              padding='same'))

    self.model.add(MaxPooling1D(pool_size=pool_size))


    self.model.add(SeparableConv1D(filters=filters * 2,
                          kernel_size=kernel_size,
                          activation='relu',
                          bias_initializer='random_uniform',
                          depthwise_initializer='random_uniform',
                          padding='same'))
    self.model.add(SeparableConv1D(filters=filters * 2,
                          kernel_size=kernel_size,
                          activation='relu',
                          bias_initializer='random_uniform',
                          depthwise_initializer='random_uniform',
                          padding='same'))
    self.model.add(GlobalAveragePooling1D())
    self.model.add(Dropout(rate=dropout_rate))
    self.model.add(Dense(1))
    self.model.add(Activation('sigmoid'))

    callbacks = []
    if self.tensorboard:
      logdir = 'logs/sepCNN_filters=' + str(filters) + "_dropout_rate="+str(dropout_rate)+"_batch_size="+str(batch_size)+"_epochs="+str(epochs)+"_kernel_size="+str(kernel_size)+"_pool_size="+str(pool_size)+"_learning_rate="+str(learning_rate)
      tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
      callbacks.append(tensorboard_callback)

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    print (self.model.summary())
    self.model.fit(tweet_padded, labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True, callbacks=callbacks)

  def predict(self, tweetsTokenized):
    if self.model is None:
      print("Please train the model first")
    else:
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


