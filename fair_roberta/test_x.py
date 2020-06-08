import torch

x = torch.load("X_xs.pt")
print(x[0]['encoding'].shape)