import pandas as pd
# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils
import en_core_web_lg
import re
import torch

roberta = RobertaModel.from_pretrained('models', checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

for size in ['xs', 's', 'm', 'l', 'xl']:
	print ("Loading data: {}".format(size))
	df = pd.read_json('../data/train_{}.jsonl'.format(size), lines=True)

	sentences = list(df['sentence'])
	option1s = list(df['option1'])
	option2s = list(df['option2'])
	answers = list(df['answer'])

	X_data = []
	y_data = []
	X_features = []

	for i, sentence in enumerate(sentences):
		print(i, end='\r')
		sentence1 = sentence.replace('_', option1s[i])
		sentence2 = sentence.replace('_', option2s[i])
		sentence1 = re.sub(r' +', ' ', sentence1)
		sentence2 = re.sub(r' +', ' ', sentence2)
		X_data.append(sentence1)
		X_data.append(sentence2)
		if answers[i] == 1:
			y_data.append([0.0, 1.0])
			y_data.append([1.0, 0.0])
		else:
			y_data.append([1.0, 0.0])
			y_data.append([0.0, 1.0])

		X_features.append(roberta.extract_features_aligned_to_words(sentence1)[0].vector)
		X_features.append(roberta.extract_features_aligned_to_words(sentence2)[0].vector)

	torch.save(torch.stack(X_features, dim=0), "X_{}.pt".format(size))
	torch.save(torch.tensor(y_data), "y_{}.pt".format(size))
	print('Saved to X_{}.pt, y_{}.pt'.format(size, size))

	