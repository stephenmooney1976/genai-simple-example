#!/usr/bin/env python3

import pandas as pd

df_input = pd.read_csv('abcnews-date-text.csv.gz', low_memory=False)
df_random = df_input.sample(frac=0.005, random_state=42)

df_random.to_csv('abcnews-date-text-small.csv.gz', index=False)
