import torch

for trainsize in ['xs', 's', 'm', 'l', 'xl']:
    X_data = torch.load("X_train_gcn_offsets_{}.bin".format(trainsize))
    y_data = torch.load("y_train_{}.bin".format(trainsize))
    print ("X size: {}, y size: {}".format( len(X_data), len(y_data) ))