#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:55:47 2023

@author: Stephen Mooney
"""

"""
Note: this is a work in progress - use at your own risk.
"""

#
# import necessary libraries
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
  dataset = dataset.shuffle(2048).batch(batch_size).prefetch(buffer_size=25000)
  return dataset

"""
"""
#
# hyperparameters
output_embedding_size = 64
lstm_size = 100

batch_size = 2**4
epochs = 150
learning_rate = 0.003

#
# Gather input training data
# This contains data of news headlines published over a period of nineteen years.
# Sourced from the reputable Australian news source ABC (Australian Broadcasting Corporation)
# Agency Site: (http://www.abc.net.au)
input_filename = 'data/abcnews-date-text-small.csv.gz'
df_input = pd.read_csv(input_filename)

# remove rows that have the same headline that is a substring of entire title
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

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.1, random_state=42)

train_ds = make_dataset(train_X, train_y, batch_size=batch_size)
val_ds = make_dataset(val_X, val_y, batch_size=batch_size)

#
# choose a generative architecture. for this I am setting up an RNN with an LSTM layer, with a output layer using softmax
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, output_embedding_size, input_length=sequence_length-1),
    tf.keras.layers.LSTM(lstm_size, return_sequences=True),
    tf.keras.layers.LSTM(lstm_size),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

#optimizer='adam'
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)

model.compile(loss=loss_fn, optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
model.summary()

# train the genrative model
with tf.device('/device:GPU:0'):
  history = model.fit(train_ds,
                      epochs=epochs,
                      verbose=1)

model.save('model-output/trained_model.keras')

with open('model-output/tokenizer.pkl', 'wb') as handle:
  pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)