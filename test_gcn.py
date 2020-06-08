import torch
import numpy as np
import sys
import en_core_web_lg
from test import get_pred
from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils
from fair_roberta_train import ClassificationHead
from dgl.data.utils import load_graphs 

LABEL_CORRECT = 1
LABEL_INCORRECT = 0
MODEL_NAME_FORMAT = "{}_model_{}"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def run(model_name_param):
    y_pred = []

    for trainsize in ['xs', 's', 'm', 'l', 'xl']:
        pred = []
        model_name = MODEL_NAME_FORMAT.format(model_name_param, trainsize)
        model = torch.load('saved_models/gcn_model_{}.pt'.format(trainsize), map_location=device)
        model.eval()

        # Get input data
        cls_tokens = torch.load('data/X_train_cls_tokens_{}.bin'.format(trainsize))
        gcn_offsets = torch.load("data/X_train_gcn_offsets_{}.bin".format(trainsize))
        all_graphs, _ = load_graphs("data/X_train_graphs_{}.bin".format(trainsize))

        # Get labels
        y_data = torch.load('data/y_{}.pt'.format(trainsize))

        y_pred.append(pred)

    # Transpose matrix
    y_pred = list(map(list, zip(*y_pred)))

    with open("test_results.txt", "w") as f:
        for row in y_pred:
            f.write(",".join([str(n) for n in row]) + "\n")

run(sys.argv[1])
