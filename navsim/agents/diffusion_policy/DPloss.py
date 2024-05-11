from typing import Dict
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from navsim.agents.diffusion_policy.DPconfig import DPConfig

def dp_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: DPConfig
):
    """
    Helper function calculating complete loss of Diffusion Policy
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Diffusion Policy config
    :return: combined loss value
    """

    trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    
    return trajectory_loss
