import torch
import pandas as pd
from transformers import DistilBertTokenizer
import numpy as np
from train import get_model
import sys

LABEL_CORRECT = 1
LABEL_INCORRECT = 0
MODEL_NAME_FORMAT = "{}_model_{}"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# model = torch.load(MODEL_NAME, map_location=device)
# model.eval()

def run(model_name_param):
    df = pd.read_json('./data/dev.jsonl', lines=True)

    # X_data = []
    # for index, row in df.iterrows():
    #     X_data.append(row['sentence'].replace('_', row['option1'] + " [SEP] "))
    #     X_data.append(row['sentence'].replace('_', row['option2'] + " [SEP] "))

    # X_test = torch.tensor([tokenizer.encode(d, pad_to_max_length="True") for d in X_data])
    # dataset = TestDataset(X_test)
    # loader = DataLoader(dataset)

    y_pred = []

    tokenizer, _ = get_model(model_name_param)

    for trainsize in ['xs', 's', 'm', 'l', 'xl']:
        pred = []
        model_name = MODEL_NAME_FORMAT.format(model_name_param, trainsize)
        model = torch.load(model_name, map_location=device)
        model.eval()
        for index, row in df.iterrows():
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

                if pred1 == 0: # option 1 is more wrong, so predict 2
                    pred.append(2) if diff1 > diff2 else pred.append(1)
                else: # option 1 is more right, so predict 1
                    pred.append(1) if diff1 > diff2 else pred.append(2)
            else:
                if pred1 == 1:
                    pred.append(1)
                elif pred2 == 1:
                    pred.append(2)

        y_pred.append(pred)

    # Transpose matrix
    y_pred = list(map(list, zip(*y_pred)))

    with open("test_results.txt", "w") as f:
        for row in y_pred:
            f.write(",".join([str(n) for n in row]) + "\n")

run(sys.argv[1])