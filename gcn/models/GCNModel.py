import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import autograd

import dgl
# from dgl import DGLGraph
# from dgl.data import MiniGCDataset
import dgl.function as fn
# from dgl.data.utils import load_graphs

# import numpy as np
# import pandas as pd

# import spacy
# import collections

# import os

# Model structure designed from: https://github.com/ianycxu/RGCN-with-BERT

class RGCNLayer(nn.Module):
    def __init__(self, feat_size, num_rels, activation=None, gated = True):

        super(RGCNLayer, self).__init__()
        self.feat_size = feat_size
        self.num_rels = num_rels
        self.activation = activation
        self.gated = gated

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, 256))
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,gain=nn.init.calculate_gain('relu'))

        if self.gated:
            self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, 1))
            nn.init.xavier_uniform_(self.gate_weight,gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, g):

        weight = self.weight
        gate_weight = self.gate_weight

        def message_func(edges):
            w = weight[edges.data['rel_type']]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            msg = msg * edges.data['norm']

            if self.gated:
                gate_w = gate_weight[edges.data['rel_type']]
                gate = torch.bmm(edges.src['h'].unsqueeze(1), gate_w).squeeze().reshape(-1,1)
                gate = torch.sigmoid(gate)
                msg = msg * gate

            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            h = self.activation(h)
            return {'h': h}
        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)

class RGCNModel(nn.Module):
    def __init__(self, h_dim, num_rels, num_hidden_layers=1, gated = True):
        super(RGCNModel, self).__init__()

        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers
        self.gated = gated

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        for _ in range(self.num_hidden_layers):
            rgcn_layer = RGCNLayer(self.h_dim, self.num_rels, activation=F.relu, gated = self.gated)
            self.layers.append(rgcn_layer)

    def forward(self, g):
        for layer in self.layers:
            layer(g) # todo: maybe g = layer(g)??

        rst_hidden = []
        for sub_g in dgl.unbatch(g):
            rst_hidden.append(  sub_g.ndata['h']   )
        return rst_hidden