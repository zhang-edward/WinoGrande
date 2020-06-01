import torch
import pandas as pd
from transformers import DistilBertTokenizer
import numpy as np

MODEL_NAME = "bert_model.mdl"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = torch.load(MODEL_NAME, map_location=device)

model.eval()

df = pd.read_json('./data/dev.jsonl', lines=True)

y_pred = []

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

for index, row in df.iterrows():
    input = row['sentence'].replace('_', row['option1'] + " [SEP] ")
    tok_input = torch.tensor(tokenizer.encode(input, pad_to_max_length="True")).unsqueeze(0)
    tok_input = tok_input.to(device)
    output = model(tok_input)
    output_logits_1 = output[0].detach().numpy()

    input = row['sentence'].replace('_', row['option2'] + " [SEP] ")
    tok_input = torch.tensor(tokenizer.encode(input, pad_to_max_length="True")).unsqueeze(0)
    tok_input = tok_input.to(device)
    output = model(tok_input)
    output_logits_2 = output[0].detach().numpy()

    pred1 = np.argmax(output_logits_1)
    pred2 = np.argmax(output_logits_2)

    if (pred1 == pred2):
        # we have a problem
        diff1 = abs(output_logits_1[0][0] - output_logits_1[0][1])
        diff2 = abs(output_logits_2[0][0] - output_logits_2[0][1])

        if diff1 > diff2:
            y_pred.append(1)
        else:
            y_pred.append(2)
    else:
        if pred1 == 1:
            y_pred.append(1)
        elif pred2 == 1:
            y_pred.append(2)

    print(y_pred[-1])

    # X_data.append(row['sentence'].replace('_', row['option1'] + " [SEP] "))
    # X_data.append(row['sentence'].replace('_', row['option2'] + " [SEP] "))
    # if row['answer'] == 1:
    #     y_data.append(1)
    #     y_data.append(0)
    # else:
    #     y_data.append(0)
    #     y_data.append(1)

# print('Number of testing examples: {:,}\n'.format(df.shape[0]))
# X_train = torch.tensor([tokenizer.encode(d, pad_to_max_length="True") for d in X_data])
# y_train = torch.tensor(y_data)

# print(X_train.shape)
# print(y_train.shape)

# y_pred = []

