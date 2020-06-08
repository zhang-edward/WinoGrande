from train_model import train_model
import torch
import torch.nn as nn
import pickle

'''
  (classifier): RobertaClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (out_proj): Linear(in_features=768, out_features=2, bias=True)
'''
class ClassificationHead(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, drop):
		super(ClassificationHead, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		
		self.dense = nn.Linear(input_dim, hidden_dim)
		self.drop = nn.Dropout(drop)
		self.out_dense = nn.Linear(hidden_dim, output_dim)
		
	def forward(self, x):
		out = self.dense(x)
		out = self.drop(out)
		out = self.out_dense(out)
		return out, "foo"

if __name__ == '__main__':
	model = ClassificationHead(1024, 768, 2, 0.1)

	for size in ['xs', 's', 'm', 'l', 'xl']:
		X_data = torch.load("fair_roberta/X_{}.pt".format(size))
		y_data = torch.load("fair_roberta/y_{}.pt".format(size))

		X_data = [x['encoding'] for x in X_data]
		X_data = torch.stack(X_data).squeeze(1)
		train_model(model, X_data, y_data, "fair_model_{}.pt".format(size))
