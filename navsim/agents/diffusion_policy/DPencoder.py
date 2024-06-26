from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import math

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone


class TranfuserEncoder(nn.Module):
    def __init__(self,feature_dim,config: TransfuserConfig):
        super().__init__()
        self._config = config
        self._backbone = TransfuserBackbone(config)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)
        self._past_traj_encoding = nn.Linear(3,config.tf_d_model)
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self.feature_dim = feature_dim
        self.output_shape=[feature_dim]
        # 新增一个全连接层来转换最终的特征维度
        self._feature_transform = nn.Linear(65 * config.tf_d_model, feature_dim)
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        camera_feature: torch.Tensor = features["camera_feature"] # (B*T, 3, 256, 1024)
        lidar_feature: torch.Tensor = features["lidar_feature"] # (B*T, 1, 256, 256)
        status_feature: torch.Tensor = features["status_feature"] # (B, 8)
        past_feature: torch.Tensor = features["past_trajectory"] # (B, T, 3)
        B = status_feature.shape[0]
        T = camera_feature.shape[0]//B
        past_feature = past_feature.view([B*T,-1])
        
         
        camera_feature = camera_feature.view([B*T,3,256,1024])
        lidar_feature = lidar_feature.view([B*T,1,256,256])
        _, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        # print("bev_feature shape",bev_feature.shape)
        #print("1past_feature shape",past_feature.shape)
        #status_encoding = self._status_encoding(status_feature)
        past_encoding = self._past_traj_encoding(past_feature)
        #print("2past_encoding shape",past_encoding.shape)
        # print("before bev_downscale",bev_feature.shape)
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        # print("bev_feature shape",bev_feature.shape)
        
        # 扩展 status_feature 以匹配时间步
        status_encoding = past_encoding
        #print("status_encoding shape",status_encoding.shape)
        feature = torch.cat([bev_feature, status_encoding[:, None]], dim=1)
        # print("feature shape",feature.shape)
        feature= self._feature_transform(feature.flatten(1))
        return feature