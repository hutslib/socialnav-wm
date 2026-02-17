#!/usr/bin/env python3

"""
Falcon-compatible World Model Policy V2
保留Falcon原始架构，World Model作为可选增强模块
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, TYPE_CHECKING
from gym import spaces
import numpy as np

from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ddppo.policy.resnet_policy import (
    ResNetEncoder,
    ResNetCLIPEncoder,
)
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.world_model.models import SocialNavWorldModel
from habitat_baselines.rl.world_model import tools
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor
from habitat.tasks.nav.nav import (
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ProximitySensor,
    EpisodicCompassSensor,
    ImageGoalSensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SocialNavWMNetV2(Net):
    """
    Network V2: Falcon作为主干 + World Model作为可选增强

    特点:
    1. 使用Falcon的ResNet encoder作为主视觉编码器
    2. World Model作为可选的辅助模块
    3. 支持退化到纯Falcon (use_world_model=False)
    4. WM训练时，policy使用 Falcon features + WM features
    """

    PRETRAINED_VISUAL_FEATURES_KEY = "visual_features"
    WM_CACHED_FEATURES_KEY = "wm_cached_feature"
    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        # World Model相关参数
        use_world_model: bool = True,
        world_model: Optional[SocialNavWorldModel] = None,
        wm_fusion_mode: str = "concat",
    ):
        super().__init__()
        # ==================== 完全复用Falcon代码顺序 ====================
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self._n_prev_action = 32
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            from habitat_baselines.utils.common import get_num_actions
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action  # test

        # Only fuse the 1D state inputs. Other inputs are processed by the
        # visual encoder
        if fuse_keys is None:
            fuse_keys = observation_space.spaces.keys()
            # removing keys that correspond to goal sensors
            goal_sensor_keys = {
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid,
                ObjectGoalSensor.cls_uuid,
                EpisodicGPSSensor.cls_uuid,
                PointGoalSensor.cls_uuid,
                HeadingSensor.cls_uuid,
                ProximitySensor.cls_uuid,
                EpisodicCompassSensor.cls_uuid,
                ImageGoalSensor.cls_uuid,
                InstanceImageGoalSensor.cls_uuid,
            }
            fuse_keys = [k for k in fuse_keys if k not in goal_sensor_keys]
        self._fuse_keys_1d: List[str] = [
            k for k in fuse_keys if len(observation_space.spaces[k].shape) == 1 and k != "human_num_sensor" and k != "localization_sensor"
        ]
        if len(self._fuse_keys_1d) != 0:
            rnn_input_size += sum(
                observation_space.spaces[k].shape[0]
                for k in self._fuse_keys_1d
            )

        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            rnn_input_size += 32

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if PointGoalSensor.cls_uuid in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
            self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
            rnn_input_size += 32

        if HeadingSensor.cls_uuid in observation_space.spaces:
            input_heading_dim = (
                observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
            )
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            rnn_input_size += 32

        if ProximitySensor.cls_uuid in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[
                ProximitySensor.cls_uuid
            ].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            rnn_input_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

        for uuid in [
            ImageGoalSensor.cls_uuid,
            InstanceImageGoalSensor.cls_uuid,
        ]:
            if uuid in observation_space.spaces:
                goal_observation_space = spaces.Dict(
                    {"rgb": observation_space.spaces[uuid]}
                )
                goal_visual_encoder = ResNetEncoder(
                    goal_observation_space,
                    baseplanes=resnet_baseplanes,
                    ngroups=resnet_baseplanes // 2,
                    make_backbone=getattr(resnet, backbone),
                    normalize_visual_inputs=normalize_visual_inputs,
                )
                setattr(self, f"{uuid}_encoder", goal_visual_encoder)

                goal_visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(goal_visual_encoder.output_shape), hidden_size
                    ),
                    nn.ReLU(True),
                )
                setattr(self, f"{uuid}_fc", goal_visual_fc)

                rnn_input_size += hidden_size

        self._hidden_size = hidden_size

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = spaces.Dict(
                {
                    k: observation_space.spaces[k]
                    for k in fuse_keys
                    if len(observation_space.spaces[k].shape) == 3
                }
            )

        if backbone.startswith("resnet50_clip"):
            self.visual_encoder = ResNetCLIPEncoder(
                observation_space
                if not force_blind_policy
                else spaces.Dict({}),
                pooling="avgpool" if "avgpool" in backbone else "attnpool",
            )
            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Linear(
                        self.visual_encoder.output_shape[0], hidden_size
                    ),
                    nn.ReLU(True),
                )
        else:
            self.visual_encoder = ResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )

            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder.output_shape), hidden_size
                    ),
                    nn.ReLU(True),
                )

        # ==================== World Model增强部分 ====================
        self.use_world_model = use_world_model
        self.wm_fusion_mode = wm_fusion_mode

        if use_world_model and world_model is not None:
            self.world_model = world_model

            # WM feature dimension: use deter only (same as WMP), for stable policy input
            wm_feat_size = world_model._config.get('dyn_deter', 200)

            # WM feature fusion
            if wm_fusion_mode == "concat":
                self.wm_feature_fc = nn.Sequential(
                    nn.Linear(wm_feat_size, hidden_size // 2, bias=False),
                    nn.LayerNorm(hidden_size // 2, eps=1e-03),
                    nn.SiLU(),
                )
                self.wm_feature_fc.apply(tools.weight_init)
                rnn_input_size += hidden_size // 2

            elif wm_fusion_mode == "add":
                self.wm_feature_fc = nn.Sequential(
                    nn.Linear(wm_feat_size, hidden_size, bias=False),
                    nn.LayerNorm(hidden_size, eps=1e-03),
                    nn.SiLU(),
                )
                self.wm_feature_fc.apply(tools.weight_init)

            elif wm_fusion_mode == "attention":
                self.wm_feature_fc = nn.Sequential(
                    nn.Linear(wm_feat_size, hidden_size, bias=False),
                    nn.LayerNorm(hidden_size, eps=1e-03),
                    nn.SiLU(),
                )
                self.wm_feature_fc.apply(tools.weight_init)

                self.fusion_attention = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=4,
                    batch_first=True,
                )

            elif wm_fusion_mode == "late":
                # 后融合: WM 不进入 RNN 输入，只与 RNN 输出融合（RNN 输入仅 visual 等）
                self.wm_feature_fc = nn.Sequential(
                    nn.Linear(wm_feat_size, hidden_size, bias=False),
                    nn.LayerNorm(hidden_size, eps=1e-03),
                    nn.SiLU(),
                )
                self.wm_feature_fc.apply(tools.weight_init)
                # 注意: late 模式不增加 rnn_input_size

            else:
                # 不支持的 fusion mode，给出警告并回退
                import warnings
                warnings.warn(
                    f"Unsupported wm_fusion_mode: {wm_fusion_mode}. "
                    f"Supported modes are: 'concat', 'add', 'attention', 'late'. "
                    f"Falling back to pure Falcon mode (no World Model)."
                )
                self.use_world_model = False
                self.world_model = None

            # 只有在真正使用 WM 时才初始化 rssm_state
            if self.use_world_model:
                self.rssm_state = None
        else:
            self.world_model = None
        # ==================== 继续Falcon原始代码 ====================

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # ==================== 完全复用Falcon代码顺序 ====================
        x = []
        aux_loss_state = {}
        wm_feat_for_late = None  # 后融合时在 RNN 之后与 out 融合
        if not self.is_blind:
            # We CANNOT use observations.get() here because self.visual_encoder(observations)
            # is an expensive operation. Therefore, we need `# noqa: SIM401`
            if (  # noqa: SIM401
                SocialNavWMNetV2.PRETRAINED_VISUAL_FEATURES_KEY
                in observations
            ):
                visual_feats = observations[
                    SocialNavWMNetV2.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                visual_feats = self.visual_encoder(observations)

            visual_feats = self.visual_fc(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats

            # ==================== World Model增强 (可选) ====================
            # 如果使用WM，对visual features进行增强
            if self.use_world_model and self.world_model is not None:
                batch_size = prev_actions.shape[0]
                cached_wm_feat = observations.get(
                    self.WM_CACHED_FEATURES_KEY, None
                )
                # WMP-style handling:
                # - Keep a persistent RSSM state only for rollout-time forward.
                # - Use a temporary state during PPO update forward to avoid
                #   mixing rollout batch (num_envs) and update batch (T * num_envs).
                use_persistent_rssm = rnn_build_seq_info is None
                use_cached_wm_feat = (
                    (cached_wm_feat is not None)
                    and ((not use_persistent_rssm) or (not self.training))
                )
                if use_persistent_rssm:
                    if (
                        self.rssm_state is None
                        or self.rssm_state["stoch"].shape[0] != batch_size
                        or masks.sum() < batch_size
                    ):
                        self.rssm_state = self.world_model.dynamics.initial(batch_size)
                    rssm_state = self.rssm_state
                else:
                    rssm_state = self.world_model.dynamics.initial(batch_size)

                if use_cached_wm_feat:
                    # Cached feature is raw RSSM feature from rollout storage.
                    # Keep wm_feature_fc in PPO graph so policy-side WM fusion
                    # layers are still trained during policy update.
                    wm_feat = cached_wm_feat.detach()
                    wm_feat_processed = self.wm_feature_fc(wm_feat)
                else:
                    # Encode with WM encoder
                    # WM is always decoupled from PPO loss.
                    with torch.no_grad():
                        wm_embed = self.world_model.encoder(observations)

                    # Convert prev_actions to one-hot
                    if self.discrete_actions:
                        action_one_hot = torch.zeros(
                            batch_size,
                            self.prev_action_embedding.num_embeddings - 1,
                            device=prev_actions.device,
                        )
                        valid_actions = (prev_actions >= 0).squeeze(-1)
                        if valid_actions.any():
                            action_one_hot[valid_actions] = F.one_hot(
                                prev_actions[valid_actions].squeeze(-1),
                                num_classes=action_one_hot.shape[1],
                            ).float()
                    else:
                        action_one_hot = prev_actions

                    # RSSM update (always no-grad for policy path).
                    with torch.no_grad():
                        if self.training:
                            embed_seq = wm_embed.unsqueeze(
                                1
                            )  # [batch, 1, embed_dim]
                            action_seq = action_one_hot.unsqueeze(
                                1
                            )  # [batch, 1, num_actions]
                            is_first = (~masks).float()  # [batch, 1]

                            post, _ = self.world_model.dynamics.observe(
                                embed_seq, action_seq, is_first, rssm_state
                            )
                            rssm_state = {k: v[:, 0] for k, v in post.items()}
                        else:
                            rssm_state = self.world_model.dynamics.img_step(
                                rssm_state, action_one_hot
                            )

                    # Get WM feature: deter only (deterministic, stable for policy)
                    wm_feat = self.world_model.dynamics.get_deter_feat(rssm_state)
                    wm_feat = wm_feat.detach()
                    wm_feat_processed = self.wm_feature_fc(wm_feat)

                if use_persistent_rssm:
                    self.rssm_state = rssm_state

                aux_loss_state["wm_feature"] = wm_feat.detach()
                aux_loss_state["wm_feature_for_storage"] = wm_feat.detach()

                # Feature fusion: 根据fusion模式处理
                if self.wm_fusion_mode == "concat":
                    # Concat模式: visual + wm都加入
                    x.append(visual_feats)
                    x.append(wm_feat_processed)
                elif self.wm_fusion_mode == "add":
                    # Add模式: 两者相加
                    x.append(visual_feats + wm_feat_processed)
                elif self.wm_fusion_mode == "attention":
                    # Attention模式: 用attention融合
                    features = torch.stack([visual_feats, wm_feat_processed], dim=1)
                    fused, _ = self.fusion_attention(features, features, features)
                    x.append(fused.mean(dim=1))
                elif self.wm_fusion_mode == "late":
                    # 后融合: 只把 visual 送入 RNN，WM 在 RNN 输出后再加
                    x.append(visual_feats)
                    wm_feat_for_late = wm_feat_processed
            else:
                # 不使用WM，直接使用Falcon visual features
                x.append(visual_feats)
            # ==================== 继续Falcon原始代码 ====================

        if len(self._fuse_keys_1d) != 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            x.append(fuse_states.float())

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert (
                    goal_observations.shape[1] == 3
                ), "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )

            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )

        for uuid in [
            ImageGoalSensor.cls_uuid,
            InstanceImageGoalSensor.cls_uuid,
        ]:
            if uuid in observations:
                goal_image = observations[uuid]

                goal_visual_encoder = getattr(self, f"{uuid}_encoder")
                goal_visual_output = goal_visual_encoder({"rgb": goal_image})

                goal_visual_fc = getattr(self, f"{uuid}_fc")
                x.append(goal_visual_fc(goal_visual_output))

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # The mask means the previous action will be zero, an extra dummy action
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        
        # Late fusion: 如果使用后融合模式，在 RNN 输出后添加 WM 特征
        if wm_feat_for_late is not None:
            out = out + wm_feat_for_late
            
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state


@baseline_registry.register_policy
class SocialNavWMPolicyV2(NetPolicy):
    """
    Policy V2: Falcon + World Model可选增强

    特点:
    1. use_world_model=False时完全等同于Falcon
    2. use_world_model=True时在Falcon基础上添加WM features
    3. 支持多种fusion模式
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        # World Model相关
        use_world_model: bool = False,
        world_model_config: Optional[Dict] = None,
        wm_fusion_mode: str = "concat",
        **kwargs,
    ):
        """
        Keyword arguments:
        rnn_type: RNN layer type; one of ["GRU", "LSTM"]
        backbone: Visual encoder backbone; one of ["resnet18", "resnet50", "resneXt50", "se_resnet50", "se_resneXt50", "se_resneXt101", "resnet50_clip_avgpool", "resnet50_clip_attnpool"]
        use_world_model: 是否使用World Model (False=纯Falcon)
        world_model_config: World Model配置
        wm_fusion_mode: Feature融合模式 ("concat", "add", "attention")
        """

        assert backbone in [
            "resnet18",
            "resnet50",
            "resneXt50",
            "se_resnet50",
            "se_resneXt50",
            "se_resneXt101",
            "resnet50_clip_avgpool",
            "resnet50_clip_attnpool",
        ], f"{backbone} backbone is not recognized."

        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        # Create World Model (if needed)
        world_model = None
        if use_world_model:
            if world_model_config is None:
                # Default config
                class DefaultConfig:
                    def get(self, key, default=None):
                        defaults = {
                            'dyn_stoch': 30,
                            'dyn_deter': 200,
                            'dyn_hidden': 200,
                            'dyn_discrete': 32,
                            'cnn_depth': 32,
                            'mlp_units': 256,
                            'act': 'SiLU',
                            'norm': True,
                            'num_actions': action_space.n if hasattr(action_space, 'n') else 4,
                        }
                        return defaults.get(key, default)
                world_model_config = DefaultConfig()

            world_model = SocialNavWorldModel(
                world_model_config,
                observation_space=observation_space,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

        # Create network
        net = SocialNavWMNetV2(
            observation_space=observation_space,
            action_space=action_space,  # for previous action
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            backbone=backbone,
            resnet_baseplanes=resnet_baseplanes,
            normalize_visual_inputs=normalize_visual_inputs,
            fuse_keys=fuse_keys,
            force_blind_policy=force_blind_policy,
            discrete_actions=discrete_actions,
            use_world_model=use_world_model,
            world_model=world_model,
            wm_fusion_mode=wm_fusion_mode,
        )

        super().__init__(
            net,
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Exclude cameras for rendering from the observation space.
        from collections import OrderedDict

        # Get agent name first
        agent_name = None
        if "agent_name" in kwargs:
            agent_name = kwargs["agent_name"]

        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        ignore_names = [
            sensor.uuid
            for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
        ]

        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )

        # World Model configuration
        use_world_model = False
        world_model_config = None
        wm_fusion_mode = "concat"

        wm_cfg = getattr(config.habitat_baselines, "world_model", None)
        if wm_cfg is not None:
            use_world_model = getattr(wm_cfg, "enabled", False)
            wm_fusion_mode = getattr(wm_cfg, "fusion_mode", "concat")

            if use_world_model:
                num_actions = action_space.n if hasattr(action_space, "n") else 4
                wm_config_dict = {
                    "encoder_type": getattr(wm_cfg, "encoder_type", "dreamer"),
                    "dyn_stoch": getattr(wm_cfg, "dyn_stoch", 30),
                    "dyn_deter": getattr(wm_cfg, "dyn_deter", 200),
                    "dyn_hidden": getattr(wm_cfg, "dyn_hidden", 200),
                    "dyn_discrete": getattr(wm_cfg, "dyn_discrete", 32),
                    "cnn_depth": getattr(wm_cfg, "cnn_depth", 32),
                    "mlp_units": getattr(wm_cfg, "mlp_units", 256),
                    "kernel_size": getattr(wm_cfg, "kernel_size", 4),
                    "minres": getattr(wm_cfg, "minres", 4),
                    "act": getattr(wm_cfg, "act", "SiLU"),
                    "norm": getattr(wm_cfg, "norm", True),
                    "num_actions": num_actions,
                    "visual_keys": getattr(wm_cfg, "visual_keys", None),
                    "fuse_keys_1d": getattr(wm_cfg, "fuse_keys_1d", None),
                    "human_state_goal_keys": getattr(wm_cfg, "human_state_goal_keys", None),
                    "num_humans": getattr(wm_cfg, "num_humans", 6),
                    "pred_horizon": getattr(wm_cfg, "pred_horizon", 5),
                    "use_goal_conditioning": getattr(wm_cfg, "use_goal_conditioning", True),
                    "state_goal_dim": getattr(wm_cfg, "state_goal_dim", 8),
                    # Falcon encoder options (used when encoder_type == "falcon")
                    "falcon_backbone": getattr(wm_cfg, "falcon_backbone", "resnet18"),
                    "falcon_baseplanes": getattr(wm_cfg, "falcon_baseplanes", 32),
                    "falcon_ngroups": getattr(wm_cfg, "falcon_ngroups", 16),
                    "falcon_use_projection": getattr(wm_cfg, "falcon_use_projection", False),
                    "falcon_target_dim": getattr(wm_cfg, "falcon_target_dim", 512),
                    "falcon_pretrained_path": getattr(wm_cfg, "falcon_pretrained_path", None),
                    "falcon_strict_load": getattr(wm_cfg, "falcon_strict_load", False),
                }

                class WMConfig:
                    def __init__(self, config_dict):
                        self._config = config_dict

                    def get(self, key, default=None):
                        return self._config.get(key, default)

                world_model_config = WMConfig(wm_config_dict)

        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            resnet_baseplanes=config.habitat_baselines.rl.ddppo.resnet_baseplanes,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
            use_world_model=use_world_model,
            world_model_config=world_model_config,
            wm_fusion_mode=wm_fusion_mode,
        )
