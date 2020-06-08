import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import autograd

from models.GCNModel import RGCNModel

class Head(nn.Module):
    """The MLP submodule"""
    def __init__(self, gcn_out_size: int, bert_out_size: int):
        super().__init__()
        self.bert_out_size = bert_out_size
        self.gcn_out_size = gcn_out_size

        self.fc = nn.Sequential(
            nn.BatchNorm1d(bert_out_size + gcn_out_size * 3),
            nn.Dropout(0.5),
            nn.Linear(bert_out_size + gcn_out_size * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 2), # todo: make sure 2 is fine.
        )
        for i, module in enumerate(self.fc):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if getattr(module, "weight_v", None) is not None:
                    nn.init.uniform_(module.weight_g, 0, 1)
                    nn.init.kaiming_normal_(module.weight_v)
                else:
                    nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, gcn_outputs, offsets_gcn, bert_embeddings):

        gcn_extracted_outputs = [gcn_outputs[i].unsqueeze(0).gather(1, offsets_gcn[i].unsqueeze(0).unsqueeze(2)
                                       .expand(-1, -1, gcn_outputs[i].unsqueeze(0).size(2))).view(gcn_outputs[i].unsqueeze(0).size(0), -1) for i in range(len(gcn_outputs))]

        gcn_extracted_outputs = torch.stack(gcn_extracted_outputs, dim=0).squeeze()

        embeddings = torch.cat((gcn_extracted_outputs, bert_embeddings), 1)

        return self.fc(embeddings)

class GPRModel(nn.Module):
    """The main model."""
    def __init__(self):
        super().__init__()
        self.RGCN =  RGCNModel(h_dim = 1024, num_rels = 3, gated = True)
        self.head = Head(256, 1024)  # gcn output   berthead output

    def forward(self, g, offsets_gcn, cls_token):
        gcn_outputs = self.RGCN(g)
        bert_head_outputs = cls_token
        head_outputs = self.head(gcn_outputs, offsets_gcn, bert_head_outputs)
        return head_outputs