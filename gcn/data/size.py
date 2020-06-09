import torch

for trainsize in ['xs', 's', 'm', 'l', 'xl']:
    X_data = torch.load("../data/X_train_gcn_offsets_{}.bin".format(trainsize))
    y_data = torch.load("../data/y_train_{}.pt".format(trainsize))
    print ("X size: {}, y size: {}".format(len(X_data, y_data)))