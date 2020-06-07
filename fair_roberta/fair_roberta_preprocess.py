import pandas as pd
# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils
import en_core_web_lg
import re
import pickle

roberta = RobertaModel.from_pretrained('models', checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

for size in ['xs', 's', 'm', 'l', 'xl']:
	df = pd.read_json('../data/train_{}.jsonl'.format(size), lines=True)

	sentences = list(df['sentence'])
	option1s = list(df['option1'])
	option2s = list(df['option2'])
	answers = list(df['answer'])

	X_data = []
	y_data = []
	X_features = []

	for i, sentence in enumerate(sentences):
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

		print(sentence1)
		X_features.append(roberta.extract_features_aligned_to_words(sentence1)[0])
		X_features.append(roberta.extract_features_aligned_to_words(sentence2)[0])

	pickle.dump(X_features, open("x_{}_features.pkl".format(size), 'wb'))