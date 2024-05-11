from typing import Dict
import numpy as np
import torch
import torch.nn as nn

from navsim.agents.diffusion_policy.DPconfig import DPConfig
from navsim.common.enums import StateSE2Index
#import the package related to the diffusion
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.common.normalizer import LinearNormalizer



class DPmodel(nn.Module):
    def __init__(
        self,
        config: DPConfig,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256,512,1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True
    ):
        super().__init__()
        feature_dim = config.feature_dim
        action_dim = config.action_dim
        n_obs_steps = config.input_history
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = feature_dim * n_obs_steps
        model = ConditionalUnet1D(
            input_dim=input_dim, # 动作
            local_cond_dim=None,
            global_cond_dim=global_cond_dim, # 接入全局的条件obs_feature_dim * n_obs_steps，n步的观察
            diffusion_step_embed_dim=diffusion_step_embed_dim, 
            down_dims=down_dims, #Unet downsampling的尺寸变化
            kernel_size=kernel_size,
            n_groups=n_groups, #控制group normalization的尺寸
            cond_predict_scale=cond_predict_scale
        )
        self.model=model
        self.noise_scheduler=config.noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = config.horizon
        self.obs_feature_dim = feature_dim
        self.action_dim = action_dim
        self.n_action_steps = config.output_length
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        camera_feature = features["camera_feature"]
        status_feature = features["status_feature"]

        # Normalize features
        camera_feature = self.normalizer.normalize(camera_feature)
        status_feature = self.normalizer.normalize(status_feature)

        # Combine features into a single global condition if needed
        if self.obs_as_global_cond:
            global_condition = torch.cat([camera_feature.flatten(start_dim=1), status_feature], dim=1)
        else:
            global_condition = None

        # Generate initial random trajectory for actions
        batch_size = camera_feature.size(0)
        action_shape = (batch_size, self.n_action_steps, self.action_dim)
        trajectory = torch.randn(action_shape, device=camera_feature.device)

        # Create mask for conditioning
        cond_mask = self.mask_generator.generate_mask(action_shape)

        # Initialize noise schedule
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Perform the reverse diffusion process
        for t in self.noise_scheduler.timesteps:
            trajectory[cond_mask] = features['action'][cond_mask] if 'action' in features else trajectory[cond_mask]
            model_output = self.model(trajectory, t, global_cond=global_condition)
            trajectory = self.noise_scheduler.step(model_output, t, trajectory).prev_sample

        trajectory[cond_mask] = features['action'][cond_mask] if 'action' in features else trajectory[cond_mask]
        # Unnormalize prediction
        naction_pred = trajectory[..., :self.action_dim]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Get action
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end, :]

        return {
            'action': action,
            'action_pred': action_pred
        }
        
        
        