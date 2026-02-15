#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


@dataclass
class HabitatBaselinesBaseConfig:
    pass


@dataclass
class Adapt3RVisualEncoderConfig(HabitatBaselinesBaseConfig):
    """Adapt3R visual encoder configuration"""
    backbone_type: str = "resnet18"
    hidden_dim: int = 252
    num_points: int = 512
    do_image: bool = True
    do_pos: bool = True
    do_rgb: bool = False
    finetune: bool = True
    xyz_proj_type: str = "nerf"
    clip_model: str = "RN50"
    do_crop: bool = True
    boundaries: List[List[float]] = field(default_factory=lambda: [[-5.0, -5.0, 0.0], [5.0, 5.0, 3.0]])
    lowdim_obs_keys: List[str] = field(default_factory=lambda: [
        "agent_0_pointgoal_with_gps_compass",
        "agent_0_localization_sensor"
    ])


@dataclass
class Adapt3RConfig(HabitatBaselinesBaseConfig):
    """Adapt3R policy configuration"""
    visual_encoder: Adapt3RVisualEncoderConfig = field(default_factory=Adapt3RVisualEncoderConfig)


# Register Adapt3R configs to config store
cs.store(
    group="habitat_baselines/rl/ddppo",
    name="adapt3r_visual_encoder_base",
    node=Adapt3RVisualEncoderConfig,
)

cs.store(
    group="habitat_baselines/rl/ddppo",
    name="adapt3r_base",
    node=Adapt3RConfig,
)
