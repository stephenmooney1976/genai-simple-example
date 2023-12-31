#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:32:57 2023

@author: Stephen Mooney
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""
"""

with open('model-output/model_dict.pkl', 'rb') as fh:
  model_dict = pickle.load(fh)

model = model_dict['model']
tokenizer = model_dict['tokenizer']
sequence_length = model_dict['sequence_length']

model.summary()

input_text = 'thai economy'

for _ in range(sequence_length):
  token_list = tokenizer.texts_to_sequences([input_text])[0]
  token_list = pad_sequences([token_list], maxlen=sequence_length-1, padding='pre')
  predicted = np.argmax(model.predict(token_list))

  output_word = ''

  for word, index in tokenizer.word_index.items():
    if index == predicted:
      output_word = word
      break

  input_text += " " + output_word

print(input_text)
