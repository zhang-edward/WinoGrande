import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import autograd
import dgl
from dgl import DGLGraph
from dgl.data import MiniGCDataset
import dgl.function as fn
from dgl.data.utils import load_graphs
import numpy as np
import pandas as pd
import spacy
import collections
import os

from models.MainModel import GPRModel
from utils import GPRDataset, collate

size = 'xl'
cls_tokens = torch.load('data/X_train_cls_tokens_{}.bin'.format(size))
gcn_offsets = torch.load("data/X_train_gcn_offsets_{}.bin".format(size))
all_graphs, _ = load_graphs("data/X_train_graphs_{}.bin".format(size))

y_data = torch.load('data/y_train_{}.bin'.format(size))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ", device)

train_dataset = GPRDataset(all_graphs, gcn_offsets, cls_tokens, y_data)

train_dataloader = DataLoader(
   train_dataset,
   collate_fn = collate,
   batch_size = 256,
   shuffle=True,
)

model = GPRModel()
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
# optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

lr_value = 0.0001
total_epoch = 100
def adjust_learning_rate(optimizers, epoch):
    # warm up
    if epoch < 10:
        lr_tmp = 0.00001
    else:
        lr_tmp = lr_value * pow((1 - 1.0 * epoch / 100), 0.9)
    
    if epoch > 36:
        lr_tmp =  0.000015 * pow((1 - 1.0 * epoch / 100), 0.9)
    
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_tmp

    return lr_tmp

save_model_name = "saved_models/gcn_model_{}.pt".format(size)

for epoch in range(total_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        with autograd.detect_anomaly():
            for i, data in enumerate(train_dataloader, 0):
                # learning rate scheduler
                lr = adjust_learning_rate([optimizer],epoch)

                graphs, gcn_offsets, cls_tokens, labels = data
                graphs, gcn_offsets, cls_tokens, labels = graphs.to(device), gcn_offsets.to(device), cls_tokens.to(device), labels.to(device)
                # inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(graphs, gcn_offsets, cls_tokens)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                print_every = 5
                if i % print_every == print_every-1:    # print every 5 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/print_every))
                    running_loss = 0.0

        torch.save(model, save_model_name)
