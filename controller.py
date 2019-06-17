import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

class Controller(nn.Module):
	def __init__(self, input_size , output_size, hidden_size ):
		super(Controller, self).__init__()
		self.fc1 = nn.Linear(input, hidden_size)
		self.fc2 = nn.Linear(input, hidden_size)
		self.reset_parameters
	
	def reset_parameters(self):
		nn.init.xavier_uniform_(self.fc1.weight, gain = 1.4)
		nn.init.xavier_uniform_(self.fc2.weight, gain = 1.4)
		nn.init.normal_(self.fc1.bias, std=0.01)
		nn.init.normal_(self.fc2.bias, std=0.01)

	
       
		
	def forward(self, x, last_read):
		x = torch.cat((x, last_read), dim = 1)
		x = F.sigmoid(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		return x
