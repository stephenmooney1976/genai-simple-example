#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:55:47 2023

@author: Stephen Mooney
"""

#
# import necessary libraries
from collections import defaultdict
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""
"""
def make_dataset(X, y, batch_size=2**6):
  dataset = tf.data.Dataset.from_tensor_slices((X, y))
  dataset = (
    dataset
      .shuffle(2048)
      .batch(batch_size)
      .prefetch(buffer_size=2048)
  )
  return dataset

"""
"""
#
# hyperparameters
output_embedding_size = 64
lstm_size = 100

batch_size = 2**4
epochs = 150
learning_rate = 1e-3

#
# Gather input training data
# This contains data of news headlines published over a period of nineteen years.
# Sourced from the reputable Australian news source ABC (Australian Broadcasting Corporation)
# Agency Site: (http://www.abc.net.au)
input_filename = 'data/abcnews-date-text-small.csv.gz'
df_input = pd.read_csv(input_filename)

# remove rows that have the same headline
arr_input_data = df_input['headline_text'].drop_duplicates().tolist()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(arr_input_data)
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(arr_input_data)

# Prepare input and target sequences
input_sequences = list()

for seq in sequences:
  for i in range(1, len(seq)):
    input_sequences.append(seq[:i+1])

sequence_length = max([len(seq) for seq in input_sequences])
padded_sequences = pad_sequences(input_sequences, maxlen=sequence_length, padding='pre')

X = np.array(padded_sequences[:, :-1])
y = np.array(padded_sequences[:, -1])

# create a tensorflow dataset for model training
ds_train = make_dataset(X, y, batch_size=batch_size)

#
# choose a generative architecture. for this I am setting up an RNN with an LSTM layer, with a output layer using softmax
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, output_embedding_size, input_length=sequence_length-1),
    tf.keras.layers.LSTM(lstm_size, return_sequences=True),
    tf.keras.layers.LSTM(lstm_size),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)

model.compile(loss=loss_fn, optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
model.summary()

cb_early_stopping = tf.keras.callbacks.EarlyStopping(
  monitor='loss',
  patience = 5,
  )

# train the genrative model
with tf.device('/device:GPU:0'):
  history = model.fit(ds_train,
                      epochs=epochs,
                      callbacks=[cb_early_stopping],
                      verbose=1)


# save model and tokenizer for inference
model_dict = defaultdict()
model_dict['model'] = model
model_dict['tokenizer'] = tokenizer
model_dict['sequence_length'] = sequence_length

with open('model-output/model_dict.pkl', 'wb') as handle:
  pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
