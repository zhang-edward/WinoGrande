from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
import pandas as pd
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from defs import Dataset

'''
[CLS] Ian volunteered to eat Dennis's menudo after already having a bowl because Ian [SEP] despised eating intestine. [SEP]",
-> [0.1, 0.9]

[CLS] Ian volunteered to eat Dennis's menudo after already having a bowl because Dennis [SEP] despised eating intestine. [SEP]",
-> [0.8, 0.2]

"option1": "Ian",
"option2": "Dennis",
"answer": "2"
'''

def run():
	df = pd.read_json('./data/train_xs.jsonl', lines=True)

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print("Device: ", device)

	X_data = []
	y_data = []

	for index, row in df.iterrows():
		X_data.append(row['sentence'].replace('_', row['option1'] + " [SEP] "))
		X_data.append(row['sentence'].replace('_', row['option2'] + " [SEP] "))
		if row['answer'] == 1:
			y_data.append([0, 1])
			y_data.append([1, 0])
		else:
			y_data.append([1, 0])
			y_data.append([0, 1])

	print('Number of training examples: {:,}\n'.format(df.shape[0]))

	tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
	# model = DistilBertModel.from_pretrained('distilbert-base-cased')
	model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')

	X_train = torch.tensor([tokenizer.encode(d, pad_to_max_length="True") for d in X_data])
	y_train = torch.tensor(y_data)

	print(X_train.shape)
	print(y_train.shape)

	# n = tokenizer.encode(X_data[0])
	# n = torch.tensor(n).unsqueeze(0)
	# # print(n)
	# # p = model(n)
	# # print(p[0].data[0][0] > p[0].data[0][1])

	batch_size = 32
	dataset = Dataset(X_train, y_train)
	loader = DataLoader(dataset, batch_size, shuffle=True)
	model = model.to(device)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters())

	for epoch in range(20):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(loader, 0):
		# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()
			# forward + backward + optimize
			outputs = model(inputs)

			loss = criterion(outputs[0], labels)
			loss.backward()
			optimizer.step()
			# print statistics

			if i % 1 == 0:    # print every 50 mini-batches
				running_loss += loss.item()
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
				running_loss = 0.0

		torch.save(model, "bert_model.mdl")

run()