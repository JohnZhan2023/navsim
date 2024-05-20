from dataclasses import dataclass
from typing import Any, List, Tuple, Dict

from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from navsim.agents.diffusion_policy.DPencoder import TranfuserEncoder
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
class MLPEncoder(nn.Module):
    def __init__(self, input_dim,  output_dim):
        super(MLPEncoder, self).__init__()
        self.output_shape = [output_dim]
        # 卷积层用于压缩 camera_feature
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # 减半图像尺寸
        self.pool = nn.MaxPool2d(2, 2)  # 再次减半图像尺寸
        # 计算卷积后的尺寸以适配全连接层
        # 假设 camera_feature 的原始尺寸是 [3, 256, 1024]
        self.conv_output_size = 16 * (256 // 4) * (1024 // 4)  # stride 和 pooling 各减半尺寸

        # 全连接层处理压缩后的 camera_feature
        self.fc_camera = nn.Linear(self.conv_output_size, output_dim)

        # 全连接层处理 status_feature
        self.fc_status = nn.Linear(input_dim, output_dim)  # 压缩 status_feature 到 output_dim 维

    def forward(self, x):
        # 处理 camera_feature
        camera = self.conv1(x['camera_feature'])
        camera = self.pool(camera)
        camera = F.relu(camera)
        flattened_camera = camera.view(camera.size(0), -1)
        output_camera = self.fc_camera(flattened_camera)

        # 处理 status_feature
        status = x['status_feature']
        output_status = self.fc_status(status)

        return output_camera, output_status
    def output_shape(self):
        return self.output_shape
@dataclass
class TranDPConfig:

    trajectory_sampling: TrajectorySampling = TrajectorySampling(
        time_horizon=4, interval_length=0.5
    )
    obs_as_global_condition=True
    diffusion_step_embed_dim=256
    down_dims=(256,512,1024)
    kernel_size=5
    n_groups=8
    k=1
    
    cond_predict_scale=True
    input_history=[0,1,2,3]
    output_length=8
    action_dim=3
    horizon=12
    noise_scheduler=DDPMScheduler(
        num_train_timesteps=1000,  # 定义扩散的时间步数
        beta_schedule="linear",    # 使用线性时间表
        beta_start=0.0001,         # 起始 beta 值
        beta_end=0.02              # 结束 beta 值
    )

    n_action_steps=8
    n_obs_steps=4
    feature_dim=256
    obs_as_global_cond=True
    shape_meta={"action": {"shape": (3,)}}  
    # obs_encoder=MLPEncoder(
    #     input_dim=8,          # 从配置的 feature_dim 字段
    #     #hidden_dim=(int)(feature_dim*1.5),         # 自定义隐藏层维度
    #     output_dim=feature_dim      # 输入历史长度乘以特征维度
    # )
    obs_encoder=TranfuserEncoder(feature_dim,TransfuserConfig)
    
    ############ parse for transformer ##############
    n_embd=256
    n_inner=1024
    debug_scene_level_prediction=False
    debug_scenario_decoding=False
    
    
    # image_architecture: str = "resnet34"
    # lidar_architecture: str = "resnet34"

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 10.0

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
