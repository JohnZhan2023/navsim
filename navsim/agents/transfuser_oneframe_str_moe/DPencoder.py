from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import math

from navsim.agents.transfuser_oneframe_str_moe.TranDPconfig import TranDPConfig
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone


class TranfuserEncoder(nn.Module):
    def __init__(self,feature_dim,config: TranDPConfig):
        super().__init__()
        self._config = config
        self._backbone = TransfuserBackbone(config)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)
        self._past_traj_encoding = nn.Linear(3,config.tf_d_model)
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self.feature_dim = feature_dim
        self.output_shape=[feature_dim]
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        camera_feature: torch.Tensor = features["camera_feature"] # (B*T, 3, 256, 1024)
        lidar_feature: torch.Tensor = features["lidar_feature"] # (B*T, 1, 256, 256)
        status_feature: torch.Tensor = features["status_feature"] # (B, 8)
        past_feature: torch.Tensor = features["past_trajectory"] # (B, T, 3)
        B = past_feature.shape[0]
        T = camera_feature.shape[1]
        past_feature = past_feature.view([B*T,-1])
        # print("=======================DEBUG========================")
        # print("past_feature shape",past_feature.shape)
        # print("status_feature shape",status_feature.shape)
        # print("camera_feature shape",camera_feature.shape)
        # print("lidar_feature shape",lidar_feature.shape)
        
        # then we only sample one frame for the inability of casual attention
        camera_feature = camera_feature[:,-1, ...]
        lidar_feature = lidar_feature[:,-1, ...]
        
        
         
        camera_feature = camera_feature.view([B,3,256,1024])
        lidar_feature = lidar_feature.view([B,1,256,256])
        _, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1) # (B, 64, 256)
        
        past_encoding = self._past_traj_encoding(past_feature).view([B, T, -1]) # (B, T, 256)

        status_encoding = self._status_encoding(status_feature) # (B, 1, 256)
        
        
        
        feature = torch.cat([bev_feature, status_encoding, past_encoding], dim=1) # (B, 69, 256)

        return feature