import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from memory import ReadHead, WriteHead
from controller import Controller


class NTM(nn.Module):
  def __init__(self, M, N,input_size , output_size, controller_out_dim, controller_hid_dim ):
    super(NTM, self).__init__()
    
    self.input_size = input_size
    self.outputs_size = outputs_size
    self.M = M
    self.N = N
    
    
    
    
