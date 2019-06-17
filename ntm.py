import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from memory import ReadHead, WriteHead
from controller import Controller


class NTM(nn.Module):
  def __init__(self, M, N,)
