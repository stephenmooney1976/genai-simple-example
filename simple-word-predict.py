#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:20:48 2023

@author: Stephen Mooney
"""

#
# import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#
# Gather input training data
#data = ['This is a sentence', 'This is another sentence', 'Finally, This is the last sentence']

data = [
  'national australia bank accused of bullying some',
  'universities team up to offer medical students',
  'international travel caps halved after national cabinet'
]

#
# preprocess data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(data)

input_sequences = []
for seq in sequences:
  for i in range(1, len(seq)):
    ngram_seq = seq[:i+1]
    input_sequences.append(ngram_seq)

max_sequence_length = max([len(seq) for seq in input_sequences])
padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]

#
# choose a generative architecture.
# For this I am setting up an RNN with an LSTM layer, with a output layer using softmax
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 64, input_length=max_sequence_length-1),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the genrative model
model.fit(X, y, epochs=100, verbose=1)

#use this model to predict next word(s):
seed_text = "universities"
next_words = 10

for _ in range(next_words):
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
  #predicted = model.predict_classes(token_list, verbose=0)
  predicted = np.argmax(model.predict(token_list))

  output_word = ""
  for word, index in tokenizer.word_index.items():
    if index == predicted:
      output_word = word
      break

  if output_word == seed_text.split().pop():
      break

  seed_text += " " + output_word

print(seed_text)