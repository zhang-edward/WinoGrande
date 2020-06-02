import torch
import pandas as pd
from transformers import DistilBertTokenizer
import numpy as np

MODEL_NAME_FORMAT = "bert_model_{}.mdl"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# model = torch.load(MODEL_NAME, map_location=device)
# model.eval()

df = pd.read_json('./data/dev.jsonl', lines=True)

y_pred = []

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

for trainsize in ['xs', 's', 'm', 'l', 'xl']:
    pred = []
    for index, row in df.iterrows():
        model_name = MODEL_NAME_FORMAT.format(trainsize)
        model = torch.load(model_name, map_location=device)
        model.eval()

        input = row['sentence'].replace('_', row['option1'] + " [SEP] ")
        tok_input = torch.tensor(tokenizer.encode(input, pad_to_max_length="True")).unsqueeze(0)
        tok_input = tok_input.to(device)
        output = model(tok_input)
        output_logits_1 = output[0].cpu().detach().numpy()

        input = row['sentence'].replace('_', row['option2'] + " [SEP] ")
        tok_input = torch.tensor(tokenizer.encode(input, pad_to_max_length="True")).unsqueeze(0)
        tok_input = tok_input.to(device)
        output = model(tok_input)
        output_logits_2 = output[0].cpu().detach().numpy()

        pred1 = np.argmax(output_logits_1)
        pred2 = np.argmax(output_logits_2)

        if (pred1 == pred2):
            # we have a problem
            diff1 = abs(output_logits_1[0][0] - output_logits_1[0][1])
            diff2 = abs(output_logits_2[0][0] - output_logits_2[0][1])

            if diff1 > diff2:
                pred.append(1)
            else:
                pred.append(2)
        else:
            if pred1 == 1:
                pred.append(1)
            elif pred2 == 1:
                pred.append(2)

    y_pred.append(pred)

# Transpose matrix
y_pred = list(map(list, zip(*y_pred)))

with open("test_results.txt", "w") as f:
    for row in f:
        f.write(",".join([str(n) for n in row]))
