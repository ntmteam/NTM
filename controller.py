import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

class Controller(nn.Module):
	def __init__(self, input , output, hid ):
		super(Controller, self).__init__()
		self.fc1 = nn.Linear(input, hid)
		self.fc2 = nn.Linear(input, hid )
		
	def forward(self, x, last_read):
		x = torch.cat((x, last_read), dim = 1)
		x = F.sigmoid(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		return x
