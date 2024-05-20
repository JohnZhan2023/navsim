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


import torch.nn.functional as F
from einops import rearrange, reduce

from navsim.common.dataclasses import Scene

from navsim.agents.diffusion_policy.DPconfig import DPConfig
from navsim.agents.diffusion_policy.DPcallback import DPCallback
from navsim.agents.diffusion_policy.DPloss import dp_loss
from navsim.agents.diffusion_policy.DP_features import DPTargetBuilder,DPFeatureBuilder
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply


class SinusoidalPosEmb(nn.Module):
    """
    Sin positional embedding, where the noisy time step are encoded as an pos embedding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
def uniform_positional_embedding(key_point_num, feat_dim):
    point_num = key_point_num
    position = torch.tensor([[6 * (point_num - i)] for i in range(point_num)])

    # Create a table of divisors for the even indices
    div_term = torch.exp(torch.arange(0, (feat_dim // 2) + (feat_dim % 2), dtype=torch.float32) * (-math.log(100.0) / max((feat_dim // 2) - 1, 1)))

    # Generate the positional encodings
    pos_embedding = torch.zeros((point_num, feat_dim))
    pos_embedding[:, 0::2] = torch.sin(position * div_term[:feat_dim // 2 + feat_dim % 2])

    if feat_dim % 2 == 0:
        pos_embedding[:, 1::2] = torch.cos(position * div_term[:feat_dim // 2])
    else:
        # For odd feat_dim, apply cosine to all but the last one which uses sine
        pos_embedding[:, 1::2] = torch.cos(position * div_term[:-1])

    return pos_embedding


class DPAgent(AbstractAgent):
    
    def __init__(
        self,
        config: DPConfig,
        lr: float,
        checkpoint_path: str = None,
    ):
        super().__init__()
        self._config = config
        self._lr=lr
        self._checkpoint_path = checkpoint_path
        self.action_shape=config.shape_meta["action"]["shape"]
        assert len(self.action_shape)==1
        action_dim=self.action_shape[0]
        obs_feature_dim=config.obs_encoder.output_shape[0]
        input_dim=action_dim+obs_feature_dim
        global_cond_dim=None
        if config.obs_as_global_condition:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim*config.n_obs_steps
        model=ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            cond_predict_scale=config.cond_predict_scale,
            
        )
        # position_embedding = uniform_positional_embedding(config.n_obs_steps, config.feature_dim).unsqueeze(0)
        # self.position_embedding = torch.nn.Parameter(position_embedding, requires_grad=True)
        # traj_position_embedding = uniform_positional_embedding(config.horizon, config.action_dim).unsqueeze(0)
        # self.traj_position_embedding = torch.nn.Parameter(traj_position_embedding, requires_grad=True)
        self.model=model
        self.obs_encoder=config.obs_encoder
        self.noise_scheduler = config.noise_scheduler
        self.mask_generator=LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if config.obs_as_global_condition else obs_feature_dim,
            max_n_obs_steps=config.n_obs_steps,
            fix_obs_steps=True,
            action_visible= True,
        )
        self.normalizer = LinearNormalizer()
        self.horizon=config.horizon
        self.obs_feature_dim=obs_feature_dim
        self.action_dim=action_dim
        self.n_action_steps=config.n_action_steps
        self.n_obs_steps=config.n_obs_steps
        self.obs_as_global_cond=config.obs_as_global_cond
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        #print("initializing human agent...")

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
    
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            # 1.5 add the time embedding
            # trajectory+=self.traj_position_embedding
            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory
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
        if self.obs_as_global_cond:
            # condition through global feature
            # print("shape of feature",nobs["camera_feature"].shape)
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            # print(this_nobs["lidar_feature"].shape)
            # this_nobs['lidar_feature']=this_nobs['lidar_feature'].unsqueeze(1)
            nobs_features = self.obs_encoder(this_nobs)
            # nobs_features = nobs_features.reshape(B, To, -1)
            # nobs_features = nobs_features + self.position_embedding
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            # we add the condition during impainting too 
            cond_data[:,:To] = nobs['past_trajectory']
            cond_mask[:,:To] = True
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
        
        # run sampling
        nsample = self.conditional_sample(
            cond_data, # cond_data的意思是有条件下生成的data！！！！！！！！！
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            )
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['past_trajectory'].unnormalize(naction_pred)
        # get action
        #为了保证Unet可以downsampling实际会输出一个更长的horizon
        start = To 
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        # print("action:",action[0])
        result = {
            'trajectory': action,
            # 'action_pred': action_pred
        }
        return result

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
                # normalize input
                
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
            #nobs_features= nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            # nobs_features = nobs_features + self.position_embedding
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        # trajectory+=self.traj_position_embedding
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        # compute loss mask
        loss_mask = ~condition_mask
        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask].to(torch.float32)
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


    
    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self.model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [DPCallback(config=self._config),]
    


