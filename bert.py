from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
import pandas as pd
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from defs import Dataset

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
# print(model(torch.tensor(inp).unsqueeze(0))[0].shape)
X_data = [
	"hello [SEP] world",
	"hello [SEP] world"
]

X_train = torch.tensor([tokenizer.encode(d) for d in X_data])
y_train = torch.tensor([
	[1, 0],
	[1, 0]
])

batch_size = 32
dataset = Dataset(X_train, y_train)
loader = DataLoader(dataset, batch_size, shuffle=True)

# if torch.cuda.is_available():
#     model = model.to("cuda")
    
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(2):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(loader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')
