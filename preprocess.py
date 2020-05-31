import pandas as pd
import numpy as np

df = pd.read_json('./data/train_xs.jsonl', lines=True)

print('Number of training examples: {:,}\n'.format(df.shape[0]))
print(df.drop(['qID'], axis=1).sample(10))

