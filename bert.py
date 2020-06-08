import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
import torch
import sys
import pandas as pd
import numpy as np
from train_model import train_model

'''
[CLS] Ian volunteered to eat Dennis's menudo after already having a bowl because Ian [SEP] despised eating intestine. [SEP]",
-> [0.1, 0.9]

[CLS] Ian volunteered to eat Dennis's menudo after already having a bowl because Dennis [SEP] despised eating intestine. [SEP]",
-> [0.8, 0.2]

"option1": "Ian",
"option2": "Dennis",
"answer": "2"
'''
def get_model(model_name, freeze=True):
	if model_name.lower() == "distilbert":
		model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')
		if (freeze):
			for param in model.distilbert.parameters():
				param.requires_grad = False
		return (DistilBertTokenizer.from_pretrained('distilbert-base-cased'), model)
	elif model_name.lower() == "roberta":
		model = RobertaForSequenceClassification.from_pretrained('roberta-base')
		if (freeze):
			for param in model.roberta.parameters():
				param.requires_grad = False
		return (RobertaTokenizer.from_pretrained('roberta-base'), model)
	else:
		print("Not a valid model!")
		return None

def run(model_name):
    for trainsize in ['xs', 's', 'm', 'l', 'xl']:
        print("TRAIN SIZE: " + trainsize)
        df = pd.read_json('./data/train_{}.jsonl'.format(trainsize), lines=True)

        X_data = []
        y_data = []

        for index, row in df.iterrows():
            X_data.append(row['sentence'].replace('_', row['option1'] + " [SEP] "))
            X_data.append(row['sentence'].replace('_', row['option2'] + " [SEP] "))
            if row['answer'] == 1:
                y_data.append([0.0, 1.0])
                y_data.append([1.0, 0.0])
            else:
                y_data.append([1.0, 0.0])
                y_data.append([0.0, 1.0])

        tokenizer, model = get_model(model_name, True)

        X_train = torch.tensor([tokenizer.encode(d, pad_to_max_length="True") for d in X_data])
        y_train = torch.tensor(y_data)

        train_model(model, X_train, y_train, '{}_model_{}'.format(model_name, trainsize))

run(sys.argv[1])