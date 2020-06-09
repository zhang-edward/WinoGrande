import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import en_core_web_lg
from dgl.data.utils import load_graphs
from utils import GPRDataset, collate
from models.MainModel import GPRModel, Head
from models.GCNModel import RGCNModel, RGCNLayer


LABEL_CORRECT = 1
LABEL_INCORRECT = 0

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def get_pred(output1, output2):
	sm = torch.nn.Softmax()
	pred1 = np.argmax(sm(output1))
	pred2 = np.argmax(sm(output2))

	if (pred1 == pred2):
		# we have a problem
		diff1 = abs(output1[0] - output1[1])
		diff2 = abs(output2[0] - output2[1])

		if pred1 == 0: # option 1 is more wrong, so predict 2
			return 2 if diff1 > diff2 else 1
		else: # option 1 is more right, so predict 1
			return 1 if diff1 > diff2 else 2
	else:
		if pred1 == 1:
			return 1
		elif pred2 == 1:
			return 2

def run():

    labeled_data_size = "s" # 'dev' for validation, 'xs',...,'xl' for training

    # Get input data
    cls_tokens = torch.load('data/X_train_cls_tokens_{}.bin'.format(labeled_data_size))
    gcn_offsets = torch.load("data/X_train_gcn_offsets_{}.bin".format(labeled_data_size))
    all_graphs, _ = load_graphs("data/X_train_graphs_{}.bin".format(labeled_data_size))

    # Get labels
    y_data = torch.load('data/y_train_{}.bin'.format(labeled_data_size))
    test_dataset = GPRDataset(all_graphs, gcn_offsets, cls_tokens, y_data)

    y_pred = []
    for trainsize in ['s']: #, 's', 'm', 'l', 'xl']:
        pred = []
        model = torch.load('saved_models/gcn_model_{}.pt'.format(trainsize), map_location=device)
        model.eval()

        test_dataloader = DataLoader(
        test_dataset,
        collate_fn = collate,
        batch_size = 2,
        shuffle=False,
        )

        num_right_answers = 0
        num_wrong_answers = 0

        with torch.no_grad():
            for i, data in enumerate(test_dataloader, 0):
                graphs, gcn_offsets, cls_tokens, labels = data
                graphs, gcn_offsets, cls_tokens, labels = graphs.to(device), gcn_offsets.to(device), cls_tokens.to(device), labels.to(device)

                outputs = model(graphs, gcn_offsets, cls_tokens)

                # TODO: Do dataloader and collate to pass inputs into the model
                out1 = outputs[0].cpu()
                out2 = outputs[1].cpu()

                lbl1 = labels[0].cpu()
                lbl2 = labels[1].cpu()

                model_prediction = get_pred(out1, out2)
                true_prediction =  get_pred(lbl1, lbl2)

                if (model_prediction == true_prediction):
                    num_right_answers += 1
                else:
                    num_wrong_answers += 1


                pred.append(model_prediction)

            accuracy = num_right_answers / (num_right_answers + num_wrong_answers)
            print("TrainSize: {}, Accuracy: {}".format(trainsize, accuracy))

    y_pred.append(pred)

    # Transpose matrix
    y_pred = list(map(list, zip(*y_pred)))

    with open("test_results.txt", "w") as f:
        for row in y_pred:
            f.write(",".join([str(n) for n in row]) + "\n")

run()
