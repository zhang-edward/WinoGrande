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

from gcn.models.MainModel import GPRModel
from gcn.utils import GPRDataset, collate

size = 'xs'
cls_tokens = torch.load('data/X_train_cls_tokens_{}.bin'.format(size))
gcn_offsets = torch.load("data/X_train_gcn_offsets_{}.bin".format(size))
all_graphs, _ = load_graphs("data/X_train_graphs_{}.bin".format(size))

y_data = torch.load('data/y_{}.pt'.format(size))

train_dataset = GPRDataset(all_graphs, gcn_offsets, cls_tokens, y_data)

train_dataloader = DataLoader(
   train_dataset,
   collate_fn = collate,
   batch_size = 64,
   shuffle=True,
)

model = GPRModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
# optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

reg_lambda = 0.035

save_model_name = "saved_models/gcn_model_{}.pt".format(size)

for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        with autograd.detect_anomaly():
            for i, data in enumerate(train_dataloader, 0):
                graphs, gcn_offsets, cls_tokens, labels = data
                # inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(graphs, gcn_offsets, cls_tokens)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                print_every = 4
                if i % print_every == print_every-1:    # print every 5 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/print_every))
                    running_loss = 0.0

        torch.save(model, save_model_name)