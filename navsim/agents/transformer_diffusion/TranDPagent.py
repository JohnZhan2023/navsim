from typing import Any, List, Dict, Union
import torch
from typing import Dict

import numpy as np
import math
import torch.nn as nn
import cv2
from torchvision import transforms
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from transformers import AutoImageProcessor, Dinov2Model
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from transformer4planning.models.decoder.diffusion_decoder import DiffusionDecoder,DiffusionWrapper, TrajDiffusionModel



import torch.nn.functional as F
from einops import rearrange, reduce

from navsim.common.dataclasses import Scene

from navsim.agents.transformer_diffusion.TranDPconfig import TranDPConfig
from navsim.agents.diffusion_policy.DPcallback import DPCallback
from navsim.agents.diffusion_policy.DPloss import dp_loss
from navsim.agents.diffusion_policy.DP_features import DPTargetBuilder,DPFeatureBuilder
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply


class TranDPagent(AbstractAgent):
    
    def __init__(
        self,
        config: TranDPConfig,
        lr: float,
        checkpoint_path: str = None,
    ):
        super().__init__()
        self._config = config
        self._lr=lr
        self._checkpoint_path = checkpoint_path
        self.k=config.k
        self.out_features = config.action_dim
        self.predict_range=config.n_action_steps+config.n_obs_steps
        print("initializing human agent...")
        self.normalizer = LinearNormalizer()
        self.horizon=config.horizon
        obs_feature_dim=config.obs_encoder.output_shape[0]
        self.obs_feature_dim=obs_feature_dim
        self.action_dim=config.action_dim
        self.n_action_steps=config.n_action_steps
        self.obs_encoder=config.obs_encoder
        self.n_obs_steps=config.n_obs_steps
        self.obs_as_global_cond=config.obs_as_global_cond
        diffusion_model = TrajDiffusionModel(config,
                                                out_features=self.out_features,
                                                predict_range=self.predict_range)
        self.model = DiffusionWrapper(diffusion_model,predict_range=self.predict_range)

    def name(self) -> str:
        """Inherited, see superclass."""
        #print("we realize the human agent")
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=self._config.input_history)# the obs n steps is 3
    

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [DPTargetBuilder(config=self._config),]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [DPFeatureBuilder(config=self._config)]
    
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features["status_feature"] = features["status_feature"].unsqueeze(1)
        
        obs_dict=features
        self.normalizer.fit(obs_dict)
        nobs = self.normalizer.normalize(obs_dict)
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        # device = self.device
        # dtype = self.dtype
        dtype = torch.float32
        device = value.device

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            
        nobs_features = self.obs_encoder(this_nobs)
        pred_traj_num = self._config.n_action_steps
        traj_hidden_state = nobs_features.reshape(B, To, -1)
        self.model.training = False
        traj_logits, scores = self.model(traj_hidden_state, batch_size=B, determin=True)
        #print("the shape of generated traj:",traj_logits.shape)
        return {"trajectory":traj_logits[:,self.n_obs_steps:,:]}

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
                # normalize input
        self.model.training = True
        batch = {"trajectory": torch.cat((features["past_trajectory"], targets["trajectory"]), dim=1), "camera_feature": features["camera_feature"],"lidar_feature": features["lidar_feature"],"status_feature": features["status_feature"]}
        #batch = {"trajectory": targets["trajectory"], "camera_feature": features["camera_feature"],"lidar_feature": features["lidar_feature"],"status_feature": features["status_feature"]}
        # normalize input
        self.normalizer.fit(batch)
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch)
        
        nactions = self.normalizer['trajectory'].normalize(batch['trajectory'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        #we don't norminalize


        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            traj_hidden_state = nobs_features.reshape(batch_size,self._config.n_obs_steps ,-1).to(dtype=torch.float32)
        trajectory_label_mask = torch.ones_like(trajectory)
        trajectory_label_mask[:, :4, :] = False

        traj_loss = None
        traj_loss = self.model.train_forward(
            traj_hidden_state,
            torch.cat((features["past_trajectory"], targets["trajectory"]), dim=1).to(dtype=torch.float32),
            
        )
        traj_loss = (traj_loss* trajectory_label_mask).sum()/(trajectory_label_mask.sum()+1e-7)
        
        return traj_loss


    
    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self.model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [DPCallback(config=self._config),]
    


