#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:56:10 2018

@author: stevenchen
"""

from __future__ import print_function
import sys
import keras
from keras.datasets import reuters, imdb
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SimpleRNN
from keras.preprocessing.text import Tokenizer



"""(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)"""

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')



tokenizer = Tokenizer(num_words=1000)

num_classes = np.max(y_train) + 1



max_features = 100000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading the FUCKING data...')
"""x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
   x_test = sequence.pad_sequences(x_test, maxlen=maxlen)"""

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
#model.add(SimpleRNN(120, dropout=0.25, recurrent_dropout=0.3))
model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
