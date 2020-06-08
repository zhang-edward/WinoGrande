import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import autograd

import dgl

class GPRDataset(Dataset):
    def __init__(self, graphs, gcn_offsets, cls_tokens, labels):

        self.graphs = graphs
        self.cls_tokens = cls_tokens
        self.gcn_offsets = gcn_offsets
        self.y = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.gcn_offsets[idx], self.cls_tokens[idx], self.y[idx]

def collate(samples):

    graphs, gcn_offsets, cls_tokens, labels = map(list, zip(*samples))

    batched_graph = dgl.batch(graphs)
    offsets_gcn = torch.stack([torch.LongTensor(x) for x in gcn_offsets], dim=0)

    cls_tokens = torch.stack(cls_tokens, dim=0).squeeze()

    labels = torch.stack(labels)

    return batched_graph, offsets_gcn, cls_tokens, labels