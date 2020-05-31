from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
import pandas as pd
import numpy as np
import torch

df = pd.read_json('./data/train_xs.jsonl', lines=True)

# print('Number of training examples: {:,}\n'.format(df.shape[0]))
# print(df.drop(['qID'], axis=1).sample(10))

# print(df['sentence'].head())
'''
[CLS] Ian volunteered to eat Dennis's menudo after already having a bowl because Ian [SEP] despised eating intestine. [SEP]",
-> [0.1, 0.9]

[CLS] Ian volunteered to eat Dennis's menudo after already having a bowl because Dennis [SEP] despised eating intestine. [SEP]",
-> [0.8, 0.2]

"option1": "Ian", 
"option2": "Dennis", 
"answer": "2"
'''

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
# model = DistilBertModel.from_pretrained('distilbert-base-cased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')

# inp = tokenizer.encode("I hate cats. [SEP] Cats scare me.")
# inp = tokenizer.encode("I hate cats. [SEP] I love cats.")
print(model(torch.tensor(inp).unsqueeze(0))[0].shape)