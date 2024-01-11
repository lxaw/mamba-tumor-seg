import numpy as np
import torch
from torch import nn
from mamba_ssm import Mamba

class SegMamba(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 # to do, complete this list
                 ):
        super().__init__()

        # mamba element
        self.mamba = Mamba(
            # to do: fill in arguments
        )
    
    def forward(self,x):
        # to do
        pass

