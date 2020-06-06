import sys
import pandas as pd
import numpy as np
from train import train_model

'''
[CLS] Ian volunteered to eat Dennis's menudo after already having a bowl because Ian [SEP] despised eating intestine. [SEP]",
-> [0.1, 0.9]

[CLS] Ian volunteered to eat Dennis's menudo after already having a bowl because Dennis [SEP] despised eating intestine. [SEP]",
-> [0.8, 0.2]

"option1": "Ian",
"option2": "Dennis",
"answer": "2"
'''

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

        train_model(model_name, X_data, y_data, '{}_model_{}'.format(model_name, trainsize))

run(sys.argv[1])