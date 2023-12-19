#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:32:57 2023

@author: Stephen Mooney
"""

import pickle

with open('model-output/tokenizer.pkl', 'rb') as handle:
  tokenizer = pickle.load(handle)