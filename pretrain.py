import sys
import torch
from train import train_model

def run(model_name):
	X_data = []
	y_data = torch.load('./pretrain/masked-wiki_y.pt') 
	with open('./pretrain/masked-wiki_X.txt', 'r') as f:
		X_data = f.readlines()
	
	train_model(model_name, X_data, y_data, "{}_pretrain.pt".format(model_name), freeze_hidden_layers=False)

run(sys.argv[1])