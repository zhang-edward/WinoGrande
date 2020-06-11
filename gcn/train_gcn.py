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

print_val_loss = True

def run(size):
    cls_tokens = torch.load('data/X_train_cls_tokens_{}.bin'.format(size))
    gcn_offsets = torch.load("data/X_train_gcn_offsets_{}.bin".format(size))
    all_graphs, _ = load_graphs("data/X_train_graphs_{}.bin".format(size))

    y_data = torch.load('data/y_train_{}.bin'.format(size))

    dev_cls_tokens = torch.load('data/X_train_cls_tokens_{}.bin'.format("dev"))
    dev_gcn_offsets = torch.load("data/X_train_gcn_offsets_{}.bin".format("dev"))
    dev_all_graphs, _ = load_graphs("data/X_train_graphs_{}.bin".format("dev"))

    dev_y_data = torch.load('data/y_train_{}.bin'.format("dev"))

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

    dev_dataset = GPRDataset(dev_all_graphs, dev_gcn_offsets, dev_cls_tokens, dev_y_data)
    dev_dataloader = DataLoader(
    dev_dataset,
    collate_fn = collate,
    batch_size = 256,
    shuffle=False,
    )

    model = GPRModel()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
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
    best_val_loss = 10000
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        with autograd.detect_anomaly():
            model.to(device)
            for i, data in enumerate(train_dataloader, 0):
                # learning rate scheduler
                # adjust_learning_rate([optimizer], epoch)

                graphs, gcn_offsets, cls_tokens, labels = data
                graphs, gcn_offsets, cls_tokens, labels = graphs.to(device), gcn_offsets.to(device), cls_tokens.to(device), labels.to(device)
                # inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                l2_reg = None            
                for w in model.parameters():
                    if not l2_reg:
                        l2_reg = w.norm(2)
                    else:
                        l2_reg = l2_reg + w.norm(2) 

                outputs = model(graphs, gcn_offsets, cls_tokens)
                loss = criterion(outputs, labels) + (l2_reg * 0.05)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                print_every = 5
                if i % print_every == print_every-1:    # print every 5 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/print_every))
                    running_loss = 0.0

            if not print_val_loss:
                continue
            
            with torch.no_grad():
                running_val_loss = 0
                n_batches = 0
                for i, data in enumerate(dev_dataloader, 0):
                    graphs, gcn_offsets, cls_tokens, labels = data
                    graphs, gcn_offsets, cls_tokens, labels = graphs.to(device), gcn_offsets.to(device), cls_tokens.to(device), labels.to(device)
                    outputs = model(graphs, gcn_offsets, cls_tokens)
                    running_val_loss += criterion(outputs, labels)
                    n_batches += 1

                val_loss = running_val_loss/n_batches
                print("validation loss: {}".format(val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print("New best val loss! Saving...")
                    torch.save(model, save_model_name)

for size in ['xs', 's', 'm', 'l', 'xl']:
    run(size)
