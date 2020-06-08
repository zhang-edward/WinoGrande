import pandas as pd
# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils
import en_core_web_lg
import re
import torch

roberta = RobertaModel.from_pretrained('models', checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
roberta.to(device)

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

		encode_dict1 = {}
		encode_dict2 = {}
		with torch.no_grad():
			encode1 = roberta.encode(sentence1)
			encode2 = roberta.encode(sentence2)
			encode_dict1['tokens'] = encode1
			encode_dict2['tokens'] = encode2
			encode_dict1['encoding'] = roberta.extract_features(encode1)[:,0].cpu()
			encode_dict2['encoding'] = roberta.extract_features(encode2)[:,0].cpu()
		X_features.append(encode_dict1)
		X_features.append(encode_dict2)
		torch.cuda.empty_cache()
		if i == 5:
			break
	torch.save(X_features, "X_{}.pt".format(size))
	torch.save(torch.tensor(y_data), "y_{}.pt".format(size))
	print('Saved to X_{}.pt, y_{}.pt'.format(size, size))

	
