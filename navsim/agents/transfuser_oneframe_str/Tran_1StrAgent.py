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

import copy
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
from transformer4planning.models.decoder.base import TrajectoryDecoder
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
import copy

import torch
from typing import Tuple, Optional, Dict
from transformers import (GPT2Model, GPT2PreTrainedModel, GPT2Config)
from transformer4planning.models.decoder.base import TrajectoryDecoder
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from dataclasses import dataclass
import numpy as np


import torch.nn.functional as F
from einops import rearrange, reduce

from navsim.common.dataclasses import Scene

from navsim.agents.transfuser_oneframe_str.TranDPconfig import TranDPConfig
from navsim.agents.transfuser_oneframe_str.DPencoder import TranfuserEncoder
from navsim.agents.diffusion_policy.DPcallback import DPCallback
from navsim.agents.transfuser_oneframe_str.TranDPfeatures import DPTargetBuilder,DPFeatureBuilder
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply

class TrajectoryGPTConfig(GPT2Config):
    def __init__(self):
        super().__init__()
        
        # Initialize additional attributes to False
        attr_list = ["use_key_points", "kp_decoder_type", "separate_kp_encoder", "use_proposal",
                     "autoregressive_proposals", "selected_exponential_past",
                     "rms_norm", "residual_in_fp32", "fused_add_norm", "raster_encoder_type",
                     "vit_intermediate_size", "mean_circular_loss",
                     "camera_image_encoder"]
        for each_attr in attr_list:
            setattr(self, each_attr, False)


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
        
        self.out_features = config.action_dim
        self.predict_range=config.n_action_steps+config.n_obs_steps
        print("initializing human agent...")
        self.normalizer = LinearNormalizer()
        self.horizon=config.horizon
        obs_feature_dim=config.obs_feature_dim
        self.obs_feature_dim=obs_feature_dim
        self.action_dim=config.action_dim
        self.n_action_steps=config.n_action_steps
        
        self.n_obs_steps=config.n_obs_steps
        self.obs_as_global_cond=config.obs_as_global_cond
        
        ###the str###
        self.decoder=None
        config_GPT=TrajectoryGPTConfig()
        self.transformer = GPT2Model(config_GPT)
        self.k=config.k
        self.obs_encoder=TranfuserEncoder(feature_dim=obs_feature_dim,config=config)
        self.use_proposal = self._config.use_proposal
        self.use_key_points = self._config.use_key_points
        self.kp_decoder_type = self._config.kp_decoder_type
        self.model_parallel = False
        self.device_map = None
        self.clf_metrics = None
        self.build_decoder()
        self.initialize()
        
    def build_decoder(self):
        # load pretrained diffusion keypoint decoder
        #TODO: add diffusion decoder trained from scratch
        if self.use_proposal:
            if self.config.task == "nuplan":
                from transformer4planning.models.decoder.base import ProposalDecoderCLS
                self.proposal_decoder = ProposalDecoderCLS(self.config, proposal_num=self.use_proposal)
            elif self.config.task == "waymo":
                from transformer4planning.models.decoder.base import ProposalDecoder
                self.proposal_decoder = ProposalDecoder(self.config)

        if self.use_key_points != 'no':
            if self.kp_decoder_type == "diffusion":
                from transformer4planning.models.decoder.diffusion_decoder import KeyPointDiffusionDecoder
                self.key_points_decoder = KeyPointDiffusionDecoder(self.config)
                if self.config.key_points_diffusion_decoder_load_from is not None:
                    print(f"Now loading pretrained key_points_diffusion_decoder from {self.config.key_points_diffusion_decoder_load_from}.")
                    state_dict = torch.load(self.config.key_points_diffusion_decoder_load_from)
                    self.key_points_decoder.model.load_state_dict(state_dict)
                    print("Pretrained keypoint decoder has been loaded!")
                else:
                    print("Now initializing diffusion decoder from scratch. Training will consume lots of time.")
            elif self.kp_decoder_type == "mlp":
                from transformer4planning.models.decoder.base import KeyPointMLPDeocder
                self.key_points_decoder = KeyPointMLPDeocder(self.config)

        self.traj_decoder = TrajectoryDecoder(self._config)
        
    def _prepare_attention_mask_for_generation(self, input_embeds):
        return torch.ones(input_embeds.shape[:2], dtype=torch.long, device=input_embeds.device)

    def _prepare_position_ids_for_generation(self, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids
        
        

    def name(self) -> str:
        """Inherited, see superclass."""
        #print("we realize the human agent")
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            print("loading checkpoint...")
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
        input_embeds= self.obs_encoder(features)
        return_dict = True
        
        padding = torch.zeros(input_embeds.size(0), 8, input_embeds.size(2), dtype=input_embeds.dtype, device=input_embeds.device)
        padded_input_embeds = torch.cat([input_embeds, padding], dim=1)
        transformer_outputs = self.transformer(
            inputs_embeds=padded_input_embeds,
            attention_mask=None,
            return_dict=return_dict,
            # **kwargs
        )
        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']
        #print("transformer_outputs_hidden_state",transformer_outputs_hidden_state.shape)
        traj_logits = self.traj_decoder.forward(transformer_outputs_hidden_state)
        return {"trajectory":traj_logits}
    
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
                # normalize input
        self.transformer.training = True
        batch = {"trajectory": torch.cat((features["past_trajectory"], targets["trajectory"]), dim=1),"past_trajectory":features["past_trajectory"], "camera_feature": features["camera_feature"],"lidar_feature": features["lidar_feature"],"status_feature": features["status_feature"]}
        input_embeds = self.obs_encoder(batch)
        return_dict = True
        padding = torch.zeros(input_embeds.size(0), 8, input_embeds.size(2), dtype=input_embeds.dtype, device=input_embeds.device)
        padded_input_embeds = torch.cat([input_embeds, padding], dim=1)
        transformer_outputs = self.transformer(
            inputs_embeds=padded_input_embeds,
            attention_mask=None,
            return_dict=return_dict,
            # **kwargs
        )
        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']
        info_dict = {"pred_length":self.n_action_steps}
        trajectory_label = targets["trajectory"].to(dtype=torch.float32, device=transformer_outputs_hidden_state.device)
        loss = torch.tensor(0, dtype=torch.float32, device=transformer_outputs_hidden_state.device)
        traj_loss, traj_logits = self.traj_decoder.compute_traj_loss(transformer_outputs_hidden_state,
                                                                     trajectory_label,
                                                                     info_dict)
        loss += traj_loss
        
        return loss

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        combined_params = list(self.transformer.parameters()) + list(self.obs_encoder.parameters())+list(self.traj_decoder.parameters())
        return torch.optim.Adam(combined_params, lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [DPCallback(config=self._config),]
    


