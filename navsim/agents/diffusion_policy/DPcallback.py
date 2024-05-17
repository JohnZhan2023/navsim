import time
from typing import Any, Dict, Optional, Union
from PIL import ImageColor

import cv2
import numpy as np
import numpy.typing as npt

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils

from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2

from navsim.visualization.config import TAB_10, MAP_LAYER_CONFIG, AGENT_CONFIG
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.agents.diffusion_policy.DPconfig import DPConfig


class DPCallback(pl.Callback):
    def __init__(
        self, config: DPConfig, num_plots: int = 3, num_rows: int = 2, num_columns: int = 2
    ) -> None:

        self._config = config

        self._num_plots = num_plots
        self._num_rows = num_rows
        self._num_columns = num_columns

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, lightning_module: pl.LightningModule
    ) -> None:
        pass

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, lightning_module: pl.LightningModule
    ) -> None:
        device = lightning_module.device
        for idx_plot in range(self._num_plots):
            features, targets = next(iter(trainer.val_dataloaders))
            features, targets = dict_to_device(features, device), dict_to_device(targets, device)
            with torch.no_grad():
                predictions = lightning_module.agent.forward(features)

            features, targets, predictions = (
                dict_to_device(features, "cpu"),
                dict_to_device(targets, "cpu"),
                dict_to_device(predictions, "cpu"),
            )
            grid = self._visualize_model(features, targets, predictions)
            trainer.logger.experiment.add_image(
                f"val_plot_{idx_plot}", grid, global_step=trainer.current_epoch
            )

    def on_test_epoch_start(
        self, trainer: pl.Trainer, lightning_module: pl.LightningModule
    ) -> None:
        pass

    def on_test_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        pass

    def on_train_epoch_start(
        self, trainer: pl.Trainer, lightning_module: pl.LightningModule
    ) -> None:
        pass

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        lightning_module: pl.LightningModule,
        unused: Optional[Any] = None,
    ) -> None:
        device = lightning_module.device
        for idx_plot in range(self._num_plots):
            features, targets = next(iter(trainer.train_dataloader))
            features, targets = dict_to_device(features, device), dict_to_device(targets, device)
            with torch.no_grad():
                predictions = lightning_module.agent.forward(features)

            features, targets, predictions = (
                dict_to_device(features, "cpu"),
                dict_to_device(targets, "cpu"),
                dict_to_device(predictions, "cpu"),
            )
            grid = self._visualize_model(features, targets, predictions)
            trainer.logger.experiment.add_image(
                f"train_plot_{idx_plot}", grid, global_step=trainer.current_epoch
            )

    def _visualize_model(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # print("visualizing model",features["camera_feature"].shape)
        sliced_tensor = features["camera_feature"][:, 1, :, :, :].squeeze(1)
        camera =sliced_tensor.permute(0,  2, 3, 1).numpy()
        #bev = targets["bev_semantic_map"].numpy()
        lidar_map = features["lidar_feature"][:, -1, :, :, :].squeeze(1).numpy()
        agent_labels = targets["agent_labels"].numpy()
        agent_states = targets["agent_states"].numpy()
        trajectory = targets["trajectory"].numpy()

        #pred_bev = predictions["bev_semantic_map"].argmax(1).numpy()
        # pred_agent_labels = predictions["agent_labels"].sigmoid().numpy()
        # pred_agent_states = predictions["agent_states"].numpy()
        pred_trajectory = predictions["trajectory"].numpy()
        print("gt traj:",trajectory[0])
        print("pred traj:",pred_trajectory[0])
        plots = []
        for sample_idx in range(self._num_rows * self._num_columns):
            plot = np.zeros((256, 768, 3), dtype=np.uint8)
            plot[:128, :512] = (camera[sample_idx] * 255).astype(np.uint8)[::2, ::2]

            # plot[128:, :256] = semantic_map_to_rgb(bev[sample_idx], self._config)
            # plot[128:, 256:512] = semantic_map_to_rgb(pred_bev[sample_idx], self._config)

            # agent_states_ = agent_states[sample_idx][agent_labels[sample_idx]]
            agent_states_=None
            # pred_agent_states_ = pred_agent_states[sample_idx][pred_agent_labels[sample_idx] > 0.5]
            pred_agent_states_=None
            plot[:, 512:] = lidar_map_to_rgb(
                lidar_map[sample_idx],
                agent_states_,
                pred_agent_states_,
                trajectory[sample_idx],
                pred_trajectory[sample_idx],
                self._config,
            )

            plots.append(torch.tensor(plot).permute(2, 0, 1))

        return vutils.make_grid(plots, normalize=False, nrow=self._num_rows)


def dict_to_device(
    dict: Dict[str, torch.Tensor], device: Union[torch.device, str]
) -> Dict[str, torch.Tensor]:

    
    for key in dict.keys():
        try:
            dict[key] = dict[key].to(device)
        except:
            dict[key]=torch.stack(dict[key],dim=0).to(device)
            dict[key]=torch.transpose(dict[key],0,1)
    return dict


def semantic_map_to_rgb(
    semantic_map: npt.NDArray[np.int64], config: DPConfig
) -> npt.NDArray[np.uint8]:

    height, width = semantic_map.shape[:2]
    rgb_map = np.ones((height, width, 3), dtype=np.uint8) * 255

    for label in range(1, config.num_bev_classes):

        if config.bev_semantic_classes[label][0] == "linestring":
            hex_color = MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS]["line_color"]
        else:
            layer = config.bev_semantic_classes[label][-1][0]  # take color of first element
            hex_color = (
                AGENT_CONFIG[layer]["fill_color"]
                if layer in AGENT_CONFIG.keys()
                else MAP_LAYER_CONFIG[layer]["fill_color"]
            )

        rgb_map[semantic_map == label] = ImageColor.getcolor(hex_color, "RGB")
    return rgb_map[::-1, ::-1]


def lidar_map_to_rgb(
    lidar_map: npt.NDArray[np.int64],
    agent_states: npt.NDArray[np.float32],
    pred_agent_states: npt.NDArray[np.float32],
    trajectory: npt.NDArray[np.float32],
    pred_trajectory: npt.NDArray[np.float32],
    config: DPConfig,
) -> npt.NDArray[np.uint8]:

    gt_color, pred_color = (0, 255, 0), (255, 0, 0)
    point_size = 2
    height, width = lidar_map.shape[:2]

    def coords_to_pixel(coords):
        pixel_center = np.array([[height / 2.0, width / 2.0]])
        coords_idcs = (coords / config.bev_pixel_size) + pixel_center
        return coords_idcs.astype(np.int32)

    rgb_map = (lidar_map * 255).astype(np.uint8)
    rgb_map = 255 - rgb_map[..., None].repeat(3, axis=-1)


    # for color, agent_state_array in zip(
    #     [gt_color, pred_color], [agent_states, pred_agent_states]
    # ):
    #     for agent_state in agent_state_array:
    #         agent_box = OrientedBox(
    #             StateSE2(*agent_state[BoundingBox2DIndex.STATE_SE2]),
    #             agent_state[BoundingBox2DIndex.LENGTH],
    #             agent_state[BoundingBox2DIndex.WIDTH],
    #             1.0,
    #         )
    #         exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
    #         exterior = coords_to_pixel(exterior)
    #         exterior = np.flip(exterior, axis=-1)
    #         cv2.polylines(rgb_map, [exterior], isClosed=True, color=color, thickness=2)
            
    for color, traj in zip(
        [gt_color, pred_color], [trajectory, pred_trajectory]
    ):
        trajectory_indices = coords_to_pixel(traj[:,:2])
        for x, y in trajectory_indices:
            cv2.circle(rgb_map, (y, x), point_size, color, -1)  # -1 fills the circle

    return rgb_map[::-1, ::-1]
