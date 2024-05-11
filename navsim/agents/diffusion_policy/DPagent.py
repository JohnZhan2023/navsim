from typing import Any, List, Dict, Union
import torch
import numpy as np
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

from navsim.common.dataclasses import Scene

from navsim.agents.diffusion_policy.DPconfig import DPConfig
from navsim.agents.diffusion_policy.DPcallback import DPCallback
from navsim.agents.diffusion_policy.DPloss import dp_loss
from navsim.agents.diffusion_policy.DP_features import DPTargetBuilder,DPFeatureBuilder

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy



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
        self._dp_model=DiffusionUnetImagePolicy(
            shape_meta=config.shape_meta,
            noise_scheduler=config.noise_scheduler,
            obs_encoder=config.obs_encoder,
            horizon=config.horizon,
            n_action_steps=config.n_action_steps,
            n_obs_steps=config.n_obs_steps,
        )
        print("initializing human agent...")

    def name(self) -> str:
        """Inherited, see superclass."""
        print("we realize the human agent")
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
        return SensorConfig.build_all_sensors(include=[self._config.input_history])# the obs n steps is 3
    

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [DPTargetBuilder(config=self._config),]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [DPFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features["status_feature"] = features["status_feature"].unsqueeze(1)
        self._dp_model.predict_action(features)
        return self._dp_model.predict_action(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return dp_loss(targets, predictions, self._config)
    
    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self._dp_model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [DPCallback(config=self._config),]
    

