# we apply the simpliest mlp on lidar point cloud and the restnet on img
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

class DriveP_encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int):
        super(DriveP_encoder, self).__init__()
        self.output_shape = [output_dim]
        
        
        