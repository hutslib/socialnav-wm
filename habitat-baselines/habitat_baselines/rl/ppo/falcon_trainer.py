#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import habitat_baselines.rl.multi_agent  # noqa: F401.
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.utils import profiling_wrapper
from habitat_baselines.common import VectorEnvFactory
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    SAVE_STATE,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.evaluator import Evaluator
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    NON_SCALAR_METRICS,
    extract_scalars_from_infos,
)
from habitat_baselines.utils.timing import g_timer
from habitat_baselines.utils.wm_visualizer import (
    render_depth_comparison,
    render_trajectory_comparison,
    compose_wm_frame,
)

def contains_inf_or_nan(observations):
    for key, value in observations.items():
        if isinstance(value, (float, int)):
            # 如果是标量，检查是否为 NaN 或 inf
            if math.isinf(value) or math.isnan(value):
                print(f"Key {key} contains inf or nan: {value}")
                return True
        elif isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
            # 如果是列表、数组或张量，检查每个元素是否为 NaN 或 inf
            if isinstance(value, torch.Tensor):
                if torch.isinf(value).any() or torch.isnan(value).any():
                    print(f"Key {key} contains inf or nan in tensor")
                    return True
            elif isinstance(value, np.ndarray):
                if np.isinf(value).any() or np.isnan(value).any():
                    print(f"Key {key} contains inf or nan in numpy array")
                    return True
            else:
                for element in value:
                    if isinstance(element, (float, int)) and (math.isinf(element) or math.isnan(element)):
                        print(f"Key {key} contains inf or nan in list/tuple: {element}")
                        return True
    return False

class _WorldModelTrainModule(torch.nn.Module):
    """DDP-friendly wrapper that computes WM losses in one forward."""

    def __init__(self, world_model: torch.nn.Module):
        super().__init__()
        self.world_model = world_model

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        kl_free_bits: float,
        kl_dyn_scale: float = 0.5,
        kl_rep_scale: float = 0.1,
    ):
        # Encode observations: (B, T, ...) -> (B*T, ...)
        batch_size, seq_len = batch["actions"].shape[:2]
        flat_obs = {}
        for key, val in batch["observations"].items():
            assert val.shape[0] == batch_size and val.shape[1] == seq_len, (
                f"Observation {key} has unexpected shape {val.shape}, "
                f"expected (batch={batch_size}, time={seq_len}, ...)"
            )
            flat_obs[key] = val.reshape(batch_size * seq_len, *val.shape[2:])

        flat_embed = self.world_model.encoder(flat_obs)
        embed = flat_embed.reshape(batch_size, seq_len, -1)

        actions_one_hot = F.one_hot(
            batch["actions"].long().squeeze(-1),
            num_classes=self.world_model.dynamics._num_actions,
        ).float()

        post, prior = self.world_model.dynamics.observe(
            embed, actions_one_hot, batch["is_first"]
        )
        feat = self.world_model.dynamics.get_feat(post)

        # Depth loss.
        depth_key = next(
            (k for k in batch["observations"] if "depth" in k.lower()),
            None,
        )
        skip_feats = getattr(self.world_model.encoder, '_cached_visual_feats_ms', None)
        depth_dist = self.world_model.heads["depth"](feat, skip_feats=skip_feats)
        if depth_key is not None:
            depth_target = batch["observations"][depth_key]
            depth_loss = -depth_dist.log_prob(depth_target).mean()
        else:
            depth_loss = depth_dist.mean().mean() * 0.0

        # Human trajectory loss (force goal path active when enabled).
        traj_head = self.world_model.heads["human_traj"]
        traj_key = next(
            (k for k in batch["observations"] if "future_trajectory" in k.lower()),
            None,
        )
        human_state_goal = next(
            (
                batch["observations"][k]
                for k in batch["observations"]
                if "human_state_goal" in k
            ),
            None,
        )
        if human_state_goal is None and getattr(
            traj_head, "use_goal_conditioning", False
        ):
            human_state_goal = feat.new_zeros(
                batch_size,
                seq_len,
                traj_head.num_humans,
                traj_head.state_goal_dim,
            )
        traj_dist = traj_head(
            feat,
            human_state_goal=human_state_goal,
        )
        traj_mean = traj_dist.mean()
        if traj_key is not None:
            traj_target = batch["observations"][traj_key]
            num_humans = traj_target.shape[2]
            human_num_key = next(
                (k for k in batch["observations"] if "human_num" in k.lower()),
                None,
            )
            if human_num_key is not None:
                human_num = (
                    batch["observations"][human_num_key]
                    .squeeze(-1)
                    .long()
                    .clamp(0, num_humans)
                )
                mask = (
                    torch.arange(num_humans, device=traj_target.device)
                    < human_num.unsqueeze(-1)
                ).float()
                # Per-human mean-squared error, masked by valid humans
                sq = (traj_mean - traj_target).pow(2).mean(dim=(-2, -1))
                traj_loss = (sq * mask).sum() / mask.sum().clamp(min=1e-8)
            else:
                traj_loss = -traj_dist.log_prob(traj_target).mean()
        else:
            traj_loss = traj_mean.mean() * 0.0

        # Reward loss.
        reward_dist = self.world_model.heads["reward"](feat)
        reward_target = batch["rewards"].float().reshape(batch_size, seq_len, -1)
        if reward_target.dim() == 2:
            reward_target = reward_target.unsqueeze(-1)
        reward_loss = -reward_dist.log_prob(reward_target).mean()

        # KL with DreamerV3-style balancing.
        kl_loss, kl_value, dyn_loss, rep_loss = self.world_model.dynamics.kl_loss(
            post, prior,
            free=kl_free_bits,
            dyn_scale=kl_dyn_scale,
            rep_scale=kl_rep_scale,
        )
        kl_loss = kl_loss.mean()

        return {
            "depth_loss": depth_loss,
            "traj_loss": traj_loss,
            "reward_loss": reward_loss,
            "kl_loss": kl_loss,
        }


@baseline_registry.register_trainer(name="falcon_trainer")
class FalconTrainer(BaseRLTrainer):
    r"""Trainer class for Falcon algorithm
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    envs: VectorEnv
    _env_spec: Optional[EnvironmentSpec]

    def __init__(self, config=None):
        super().__init__(config)

        self._agent = None
        self.envs = None
        self.obs_transforms = []
        self._is_static_encoder = False
        self._encoder = None
        self._env_spec = None

        # Distributed if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def _create_obs_transforms(self):
        self.obs_transforms = get_active_obs_transforms(self.config)
        self._env_spec.observation_space = apply_obs_transforms_obs_space(
            self._env_spec.observation_space, self.obs_transforms
        )

    def _create_agent(self, resume_state, **kwargs) -> AgentAccessMgr:
        """
        Sets up the AgentAccessMgr. You still must call `agent.post_init` after
        this call. This only constructs the object.
        """

        self._create_obs_transforms()
        return baseline_registry.get_agent_access_mgr(
            self.config.habitat_baselines.rl.agent.type
        )(
            config=self.config,
            env_spec=self._env_spec,
            is_distrib=self._is_distributed,
            device=self.device,
            resume_state=resume_state,
            num_envs=self.envs.num_envs,
            percent_done_fn=self.percent_done,
            **kwargs,
        )

    def _init_envs(self, config=None, is_eval: bool = False):
        if config is None:
            config = self.config
        # print(config) ##
        env_factory: VectorEnvFactory = hydra.utils.instantiate(
            config.habitat_baselines.vector_env_factory
        )
        self.envs = env_factory.construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=is_eval,
            is_first_rank=(
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ),
        )

        self._env_spec = EnvironmentSpec(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            orig_action_space=self.envs.orig_action_spaces[0],
        )

        # The measure keys that should only be logged on rank0 and nowhere
        # else. They will be excluded from all other workers and only reported
        # from the single worker.
        self._rank0_keys: Set[str] = set(
            list(self.config.habitat.task.rank0_env0_measure_names)
            + list(self.config.habitat.task.rank0_measure_names)
        )

        # Information on measures that declared in `self._rank0_keys` or
        # to be only reported on rank0. This is seperately logged from
        # `self.window_episode_stats`.
        self._single_proc_infos: Dict[str, List[float]] = {}

    def _init_train(self, resume_state=None):
        if resume_state is None:
            resume_state = load_resume_state(self.config)

        if resume_state is not None:
            if not self.config.habitat_baselines.load_resume_state_config:
                raise FileExistsError(
                    f"The configuration provided has habitat_baselines.load_resume_state_config=False but a previous training run exists. You can either delete the checkpoint folder {self.config.habitat_baselines.checkpoint_folder}, or change the configuration key habitat_baselines.checkpoint_folder in your new run."
                )

            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )

        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        self._add_preemption_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                    local_rank
                )
                # Multiply by the number of simulators to make sure they also get unique seeds
                self.config.habitat.seed += (
                    torch.distributed.get_rank()
                    * self.config.habitat_baselines.num_environments
                )

            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
            torch.manual_seed(self.config.habitat.seed)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

        # remove the non scalar measures from the measures since they can only be used in
        # evaluation
        for non_scalar_metric in NON_SCALAR_METRICS:
            non_scalar_metric_root = non_scalar_metric.split(".")[0]
            if non_scalar_metric_root in self.config.habitat.task.measurements:
                with read_write(self.config):
                    OmegaConf.set_struct(self.config, False)
                    self.config.habitat.task.measurements.pop(
                        non_scalar_metric_root
                    )
                    OmegaConf.set_struct(self.config, True)
                if self.config.habitat_baselines.verbose:
                    logger.info(
                        f"Removed metric {non_scalar_metric_root} from metrics since it cannot be used during training."
                    )

        self._init_envs()

        self.device = get_device(self.config)

        if rank0_only() and not os.path.isdir(
            self.config.habitat_baselines.checkpoint_folder
        ):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        logger.add_filehandler(self.config.habitat_baselines.log_file)

        self._agent = self._create_agent(resume_state)
        if self._is_distributed:
            self._agent.init_distributed(find_unused_params=False)  # type: ignore
        self._agent.post_init()

        self._is_static_encoder = (
            not self.config.habitat_baselines.rl.ddppo.train_encoder
        )
        self._ppo_cfg = self.config.habitat_baselines.rl.ppo

        observations = self.envs.reset()
        observations = self.envs.post_step(observations)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        # Key Modification between the trainer and the original ppo trainer
        if self._is_static_encoder:
            self._encoder = self._agent.actor_critic.visual_encoder
            if self._encoder is None:
                self._encoder = self._agent._agents[0].actor_critic.visual_encoder
                with inference_mode():
                    batch_temp = {key.replace('agent_0_', ''): value for key, value in batch.items()}
                    batch[
                        'agent_0_' + PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch_temp)
            else:
                with inference_mode():
                    batch[
                        PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch)

        self._agent.rollouts.insert_first_observations(batch)

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self._ppo_cfg.reward_window_size)
        )
        self._resume_update_num = 0
        # logger.info(f"[INIT DEBUG] running_episode_stats initialized with "
        #              f"num_envs={self.envs.num_envs}, all zeros. "
        #              f"window_size={self._ppo_cfg.reward_window_size}")

        # ==================== World Model Training Setup ====================
        self._init_world_model_training()

        # Load WM optimizer state if resuming
        if resume_state is not None and self.train_world_model:
            self._load_wm_state(resume_state)

        self.t_start = time.time()

    def _init_world_model_training(self):
        """Initialize World Model training components"""
        # Check if WM is enabled (supports both dict-like and structured WorldModelConfig)
        wm_config = getattr(self.config.habitat_baselines, "world_model", None)
        if wm_config is None:
            self.use_world_model = False
            self.train_world_model = False
        else:
            self.use_world_model = getattr(wm_config, "enabled", False)
            self.train_world_model = self.use_world_model and getattr(
                wm_config, "train_world_model", False
            )

        if not self.use_world_model:
            logger.info("World Model is disabled")
            return

        # Get WM from policy (needed for both training and pretrained loading)
        try:
            if hasattr(self._agent.actor_critic, 'net'):
                # Single agent
                self.world_model = getattr(self._agent.actor_critic.net, 'world_model', None)
            else:
                # Multi-agent, get from first agent
                self.world_model = getattr(self._agent._agents[0].actor_critic.net, 'world_model', None)

            if self.world_model is None:
                logger.warning("World Model not found in policy, disabling WM training")
                self.train_world_model = False
                return
        except Exception as e:
            logger.warning(f"Failed to get World Model: {e}, disabling WM training")
            self.train_world_model = False
            return

        # Load pretrained World Model weights if specified
        # (even when train_world_model=False, we still need to load pretrained weights for inference)
        pretrained_wm_path = getattr(wm_config, "pretrained_wm_checkpoint", None)
        if pretrained_wm_path is not None and pretrained_wm_path != "":
            self._load_pretrained_world_model(pretrained_wm_path)

        if not self.train_world_model:
            logger.info("World Model training is disabled (pretrained WM loaded for inference only)")
            return

        logger.info("Initializing World Model training...")

        # Freeze WM encoder if specified (after loading pretrained weights)
        self.freeze_wm_encoder = getattr(wm_config, "freeze_wm_encoder", False)
        if self.freeze_wm_encoder:
            self.world_model.freeze_encoder()

        # WM training parameters (from config)
        self.wm_train_ratio = getattr(wm_config, "wm_train_ratio", 0.1)
        self.wm_warmup_updates = getattr(wm_config, "wm_warmup_updates", 1000)
        self.wm_grad_clip = getattr(wm_config, "wm_grad_clip", 100.0)
        self.wm_batch_size = getattr(wm_config, "wm_batch_size", 16)
        self.wm_sequence_length = getattr(wm_config, "wm_sequence_length", 50)
        self.wm_epochs_per_update = getattr(wm_config, "wm_epochs_per_update", 1)

        # Loss scales
        self.depth_loss_scale = getattr(wm_config, "depth_loss_scale", 1.0)
        self.traj_loss_scale = getattr(wm_config, "traj_loss_scale", 1.0)
        self.reward_loss_scale = getattr(wm_config, "reward_loss_scale", 1.0)
        self.kl_loss_scale = getattr(wm_config, "kl_loss_scale", 0.1)
        self.kl_free_bits = getattr(wm_config, "kl_free_bits", 1.0)
        self.kl_dyn_scale = getattr(wm_config, "kl_dyn_scale", 0.5)
        self.kl_rep_scale = getattr(wm_config, "kl_rep_scale", 0.1)

        # Create WM optimizer (only include trainable params, excludes frozen encoder)
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.trainable_parameters(),
            lr=getattr(wm_config, "wm_lr", 3e-4),
            eps=getattr(wm_config, "opt_eps", 1e-5),
            weight_decay=getattr(wm_config, "weight_decay", 0.0),
        )

        # Build WM train wrapper and optionally wrap with DDP.
        self.wm_train_model = _WorldModelTrainModule(self.world_model).to(
            self.device
        )
        self.wm_ddp_enabled = False
        wm_use_ddp = getattr(wm_config, "ddp", True)
        if self._is_distributed and wm_use_ddp:
            if self.device.type == "cuda":
                device_index = (
                    self.device.index if self.device.index is not None else 0
                )
                self.wm_train_model = torch.nn.parallel.DistributedDataParallel(
                    self.wm_train_model,
                    device_ids=[device_index],
                    output_device=device_index,
                    find_unused_parameters=False,
                )
            else:
                self.wm_train_model = torch.nn.parallel.DistributedDataParallel(
                    self.wm_train_model,
                    find_unused_parameters=False,
                )
            self.wm_ddp_enabled = True

        # Create replay buffer (env-wise trajectories, initialized lazily)
        buffer_size = getattr(wm_config, "replay_buffer_size", 100000)
        self.replay_buffer_size = buffer_size
        self.replay_buffer = None
        self.replay_buffer_num_envs = None
        self.replay_buffer_env_capacity = None
        self.replay_buffer_warmup = getattr(wm_config, "replay_buffer_warmup", 5000)

        logger.info(f"World Model training initialized:")
        logger.info(f"  - wm_train_ratio: {self.wm_train_ratio} (update every {int(1/self.wm_train_ratio)} policy updates)")
        logger.info(f"  - wm_warmup_updates: {self.wm_warmup_updates}")
        logger.info("  - wm_mode: decoupled (independent optimizer)")
        logger.info(f"  - freeze_wm_encoder: {self.freeze_wm_encoder}")
        logger.info(f"  - replay_buffer_size: {buffer_size}")
        logger.info(f"  - wm_batch_size: {self.wm_batch_size}")
        logger.info(f"  - wm_sequence_length: {self.wm_sequence_length}")
        logger.info(f"  - wm_ddp: {self.wm_ddp_enabled}")

        # WM is trained only by wm_optimizer, never by PPO loss.
        self._set_world_model_grad_enabled(False)
        self._clear_world_model_grads()

    def _set_world_model_grad_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "world_model") or self.world_model is None:
            return
        for param in self.world_model.parameters():
            param.requires_grad_(enabled)

    def _clear_world_model_grads(self) -> None:
        if not hasattr(self, "world_model") or self.world_model is None:
            return
        if hasattr(self, "wm_optimizer") and self.wm_optimizer is not None:
            self.wm_optimizer.zero_grad(set_to_none=True)
        for param in self.world_model.parameters():
            param.grad = None

    def _load_pretrained_world_model(self, checkpoint_path: str):
        """加载预训练的 World Model 权重

        Args:
            checkpoint_path: 预训练checkpoint的路径（可以是完整checkpoint或只包含world_model的文件）
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Pretrained WM checkpoint not found: {checkpoint_path}")
            return

        try:
            if rank0_only():
                logger.info(f"Loading pretrained World Model from: {checkpoint_path}")

            # Load checkpoint（与 save_checkpoint 一致：单 agent 为 state_dict，多 agent 为 {0: {state_dict}, 1: {...}}）
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict")
            if state_dict is None:
                for v in checkpoint.values():
                    if isinstance(v, dict) and "state_dict" in v:
                        state_dict = v["state_dict"]
                        break
            # state_dict 来自 policy.state_dict()，key 为 "net.world_model.xxx"（单 agent）或带 "module."（DDP）
            wm_prefixes = ("net.world_model.", "module.net.world_model.")
            wm_state_dict = {}
            if state_dict:
                for k, v in state_dict.items():
                    if not isinstance(k, str):
                        continue
                    for prefix in wm_prefixes:
                        if k.startswith(prefix):
                            wm_state_dict[k[len(prefix):]] = v
                            break

            if not wm_state_dict:
                logger.warning("No world_model parameters found in checkpoint")
                return

            # Load weights
            missing_keys, unexpected_keys = self.world_model.load_state_dict(
                wm_state_dict, strict=False
            )

            if rank0_only():
                loaded_keys = [k for k in wm_state_dict if k not in missing_keys]
                loaded_params = sum(wm_state_dict[k].numel() for k in loaded_keys)
                total_params = sum(p.numel() for p in self.world_model.parameters())
                logger.info(
                    f"[World Model] Successfully loaded pretrained WM checkpoint!\n"
                    f"  Source: {checkpoint_path}\n"
                    f"  Loaded layers: {len(loaded_keys)}/{len(wm_state_dict)}\n"
                    f"  Loaded params: {loaded_params:,} / {total_params:,} "
                    f"({loaded_params/total_params*100:.1f}%)"
                )
                if missing_keys:
                    logger.warning(
                        f"  Missing keys ({len(missing_keys)}, will be randomly initialized):"
                    )
                    for key in missing_keys[:10]:
                        logger.warning(f"    - {key}")
                    if len(missing_keys) > 10:
                        logger.warning(f"    ... and {len(missing_keys) - 10} more")
                if unexpected_keys:
                    logger.warning(
                        f"  Unexpected keys ({len(unexpected_keys)}, ignored):"
                    )
                    for key in unexpected_keys[:10]:
                        logger.warning(f"    - {key}")
                    if len(unexpected_keys) > 10:
                        logger.warning(f"    ... and {len(unexpected_keys) - 10} more")

        except Exception as e:
            logger.error(f"Failed to load pretrained World Model: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _load_wm_state(self, resume_state):
        """从 checkpoint 加载 World Model optimizer 状态"""
        if 'wm_optimizer_state' in resume_state:
            self.wm_optimizer.load_state_dict(resume_state['wm_optimizer_state'])
            if rank0_only():
                logger.info("Loaded WM optimizer state from checkpoint")

        if 'replay_buffer_size' in resume_state:
            buffer_size = resume_state['replay_buffer_size']
            if rank0_only():
                logger.info(f"Previous replay buffer had {buffer_size} experiences (buffer not restored)")

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            **self._agent.get_save_state(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state  # type: ignore

        # Save World Model optimizer state
        if self.train_world_model and hasattr(self, 'wm_optimizer'):
            checkpoint["wm_optimizer_state"] = self.wm_optimizer.state_dict()
            checkpoint["replay_buffer_size"] = self._get_replay_buffer_size()
            if rank0_only():
                logger.info(
                    "Saving WM optimizer state "
                    f"(replay buffer size: {self._get_replay_buffer_size()})"
                )

        save_file_path = os.path.join(
            self.config.habitat_baselines.checkpoint_folder, file_name
        )
        torch.save(checkpoint, save_file_path)
        torch.save(
            checkpoint,
            os.path.join(
                self.config.habitat_baselines.checkpoint_folder, "latest.pth"
            ),
        )
        if self.config.habitat_baselines.on_save_ckpt_callback is not None:
            hydra.utils.call(
                self.config.habitat_baselines.on_save_ckpt_callback,
                save_file_path=save_file_path,
            )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        kwargs.setdefault("weights_only", False)
        return torch.load(checkpoint_path, *args, **kwargs)

    def _should_update_world_model(self, update):
        """判断是否应该更新 World Model"""
        if not self.train_world_model:
            return False

        # 检查 warmup
        if update < self.wm_warmup_updates:
            return False

        # 检查 replay buffer 是否足够
        if self._get_replay_buffer_size() < self.replay_buffer_warmup:
            return False

        # 检查更新频率（关键逻辑：wm_train_ratio）
        update_interval = int(1 / self.wm_train_ratio)
        if update % update_interval != 0:
            return False

        return True

    def _get_replay_buffer_size(self) -> int:
        """Get total WM replay size across all env trajectories."""
        if self.replay_buffer is None:
            return 0
        return int(sum(len(env_buffer) for env_buffer in self.replay_buffer))

    def _store_rollout_to_buffer(self, rollouts):
        """将 rollout 数据存储到 replay buffer。兼容单智能体 RolloutStorage 与多智能体 MultiStorage。"""
        if not self.train_world_model:
            return

        # 多智能体下 rollouts 为 MultiStorage，无 num_steps/buffers；用第一个 agent（如 robot）的 storage
        if hasattr(rollouts, "_active_storages") and len(rollouts._active_storages) > 0:
            storage = rollouts._active_storages[0]
        else:
            storage = rollouts

        num_steps = getattr(storage, "num_steps", None)
        if num_steps is None:
            return
        buffers = getattr(storage, "buffers", None)
        if buffers is None:
            return

        obs_buf = buffers["observations"]
        actions_buf = buffers["actions"]
        rewards_buf = buffers["rewards"]
        masks_buf = buffers["masks"]
        num_envs = actions_buf.size(1)

        if self.replay_buffer is None or self.replay_buffer_num_envs != num_envs:
            # Keep global capacity roughly fixed while storing env-wise trajectories.
            per_env_capacity = max(1, self.replay_buffer_size // num_envs)
            self.replay_buffer = [deque(maxlen=per_env_capacity) for _ in range(num_envs)]
            self.replay_buffer_num_envs = num_envs
            self.replay_buffer_env_capacity = per_env_capacity
            if rank0_only():
                logger.info(
                    "Initialized env-wise WM replay buffer: "
                    f"num_envs={num_envs}, per_env_capacity={per_env_capacity}, "
                    f"total_capacity~={per_env_capacity * num_envs}"
                )

        # 存到 CPU，避免 replay buffer 撑大 GPU 显存（采样 WM batch 时会 .to(device)）
        for step in range(num_steps):
            for env_idx in range(num_envs):
                obs_dict = {}
                for key in obs_buf.keys():
                    t = obs_buf[key]
                    obs_dict[key] = t[step, env_idx].detach().clone().cpu()
                mask = masks_buf[step, env_idx].detach().clone().cpu()
                experience = {
                    "observations": obs_dict,
                    "actions": actions_buf[step, env_idx].detach().clone().cpu(),
                    "rewards": rewards_buf[step, env_idx].detach().clone().cpu(),
                    "masks": mask,
                    # is_first = not mask (当episode结束时mask=0,is_first=1)
                    "is_first": (~mask.bool()).float().cpu(),
                }
                self.replay_buffer[env_idx].append(experience)

    def _sample_wm_batch(self):
        """从 env-wise replay buffer 采样连续子序列（每个序列来自单一 env）。"""
        if self.replay_buffer is None:
            return None

        valid_env_ids = []
        env_weights = []
        for env_idx, env_buffer in enumerate(self.replay_buffer):
            num_starts = len(env_buffer) - self.wm_sequence_length + 1
            if num_starts > 0:
                valid_env_ids.append(env_idx)
                env_weights.append(num_starts)

        if len(valid_env_ids) == 0:
            return None

        env_weights = np.asarray(env_weights, dtype=np.float64)
        env_weights = env_weights / env_weights.sum()

        # 收集 sequences（每个 sequence 来自同一 env 的连续片段）
        batch_sequences = []
        for _ in range(self.wm_batch_size):
            env_idx = int(np.random.choice(valid_env_ids, p=env_weights))
            env_traj = list(self.replay_buffer[env_idx])
            max_start_idx = len(env_traj) - self.wm_sequence_length
            start_idx = int(np.random.randint(0, max_start_idx + 1))
            sequence = env_traj[start_idx : start_idx + self.wm_sequence_length]
            batch_sequences.append(sequence)

        # 转换为 tensor batch
        batch = {}

        # 处理 observations（需要特殊处理，因为是 dict）
        obs_keys = batch_sequences[0][0]['observations'].keys()
        batch['observations'] = {}
        for key in obs_keys:
            # Shape: [batch_size, sequence_length, ...]
            batch['observations'][key] = torch.stack([
                torch.stack([step['observations'][key] for step in seq])
                for seq in batch_sequences
            ]).to(self.device)

        # 处理其他数据
        for data_key in ['actions', 'rewards', 'masks', 'is_first']:
            batch[data_key] = torch.stack([
                torch.stack([step[data_key] for step in seq])
                for seq in batch_sequences
            ]).to(self.device)

        return batch

    def _get_wm_sampling_stats(self):
        """Return WM sampling stats: valid env count and avg traj length."""
        if self.replay_buffer is None:
            return 0, 0.0

        valid_env_lengths = [
            len(env_buffer)
            for env_buffer in self.replay_buffer
            if len(env_buffer) >= self.wm_sequence_length
        ]
        if len(valid_env_lengths) == 0:
            return 0, 0.0

        return len(valid_env_lengths), float(np.mean(valid_env_lengths))

    def _compute_kl_loss(self, post, prior):
        """计算 KL divergence with free bits"""
        # KL divergence between posterior and prior
        kl_loss = torch.distributions.kl.kl_divergence(
            self.world_model.dynamics.get_dist(post),
            self.world_model.dynamics.get_dist(prior)
        )

        # Apply free bits (避免 KL collapse)
        # Free bits: 只惩罚 KL > kl_free_bits 的部分
        kl_loss = torch.clamp(kl_loss - self.kl_free_bits, min=0.0)

        return kl_loss.mean()

    def _update_world_model(self, update):
        """训练 World Model"""
        self._set_world_model_grad_enabled(True)
        self.wm_train_model.train()

        if rank0_only():
            logger.info(f"[Update {update}] Updating World Model...")
            valid_env_count, avg_env_length = self._get_wm_sampling_stats()
            logger.info(
                f"[Update {update}] WM sampling stats: "
                f"valid_envs={valid_env_count}, avg_env_length={avg_env_length:.1f}"
            )

        wm_losses_sum = {
            'depth_loss': 0.0,
            'traj_loss': 0.0,
            'reward_loss': 0.0,
            'kl_loss': 0.0,
            'total_loss': 0.0
        }

        num_batches_trained = 0
        last_batch = None

        # 训练多个 epochs
        for epoch in range(self.wm_epochs_per_update):
            # 1. 从 replay buffer 采样 sequences
            batch = self._sample_wm_batch()

            # DDP safety: all ranks must make the same control-flow decision.
            if self._is_distributed and torch.distributed.is_initialized():
                has_batch = torch.tensor(
                    1 if batch is not None else 0,
                    device=self.device,
                    dtype=torch.int32,
                )
                torch.distributed.all_reduce(
                    has_batch, op=torch.distributed.ReduceOp.MIN
                )
                global_has_batch = int(has_batch.item()) == 1
                if not global_has_batch:
                    if rank0_only():
                        logger.warning(
                            f"[Update {update}] Global replay buffer too small, "
                            "skipping WM update this round"
                        )
                    break

            if batch is None:
                if rank0_only():
                    logger.warning(f"[Update {update}] Replay buffer too small, skipping WM update")
                break

            # WMP-style alignment: always treat the first step of each sampled
            # sequence as a new sequence start.
            batch['is_first'][:, 0] = 1.0

            # 2. WM forward pass (through DDP wrapper if enabled)
            wm_losses = self.wm_train_model(
                batch, self.kl_free_bits,
                kl_dyn_scale=self.kl_dyn_scale,
                kl_rep_scale=self.kl_rep_scale,
            )
            depth_loss = wm_losses["depth_loss"]
            traj_loss = wm_losses["traj_loss"]
            reward_loss = wm_losses["reward_loss"]
            kl_loss = wm_losses["kl_loss"]

            # 4. Total WM loss
            wm_loss = (
                depth_loss * self.depth_loss_scale +
                traj_loss * self.traj_loss_scale +
                reward_loss * self.reward_loss_scale +
                kl_loss * self.kl_loss_scale
            )

            # 5. Backward and optimizer step
            self.wm_optimizer.zero_grad()
            wm_loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.world_model.parameters(),
                self.wm_grad_clip
            )

            self.wm_optimizer.step()

            # 累积 losses
            wm_losses_sum['depth_loss'] += depth_loss.item()
            wm_losses_sum['traj_loss'] += traj_loss.item()
            wm_losses_sum['reward_loss'] += reward_loss.item()
            wm_losses_sum['kl_loss'] += kl_loss.item()
            wm_losses_sum['total_loss'] += wm_loss.item()
            num_batches_trained += 1
            last_batch = batch

        # Keep WM optimizer path isolated from PPO path.
        self._clear_world_model_grads()
        self._set_world_model_grad_enabled(False)

        # 平均 losses
        if num_batches_trained > 0:
            for key in wm_losses_sum:
                wm_losses_sum[key] /= num_batches_trained

        if rank0_only():
            logger.info(
                f"[Update {update}] WM training completed: "
                f"depth={wm_losses_sum['depth_loss']:.3f}, "
                f"traj={wm_losses_sum['traj_loss']:.3f}, "
                f"reward={wm_losses_sum['reward_loss']:.3f}, "
                f"kl={wm_losses_sum['kl_loss']:.3f}, "
                f"total={wm_losses_sum['total_loss']:.3f}"
            )

        # Visualize a few samples from the last batch
        if rank0_only() and last_batch is not None:
            self._last_wm_vis_batch = last_batch

        return wm_losses_sum

    @torch.no_grad()
    def _visualize_wm_predictions(self, writer, step, num_samples=3):
        """Generate WM prediction visualizations, save to disk and optionally to tensorboard."""
        import cv2

        batch = getattr(self, '_last_wm_vis_batch', None)
        if batch is None:
            return

        wm = self.world_model
        wm.eval()

        B, T = batch["actions"].shape[:2]
        num_samples = min(num_samples, B)

        flat_obs = {}
        for key, val in batch["observations"].items():
            flat_obs[key] = val.reshape(B * T, *val.shape[2:])

        flat_embed = wm.encoder(flat_obs)
        embed = flat_embed.reshape(B, T, -1)

        actions_oh = F.one_hot(
            batch["actions"].long().squeeze(-1),
            num_classes=wm.dynamics._num_actions,
        ).float()

        post, _ = wm.dynamics.observe(embed, actions_oh, batch["is_first"])
        feat = wm.dynamics.get_feat(post)

        skip_feats = getattr(wm.encoder, '_cached_visual_feats_ms', None)
        depth_dist = wm.heads["depth"](feat, skip_feats=skip_feats)
        pred_depth = depth_dist.mean()

        human_state_goal = next(
            (batch["observations"][k] for k in batch["observations"] if "human_state_goal" in k),
            None,
        )

        traj_dist = wm.heads["human_traj"](feat, human_state_goal=human_state_goal)
        pred_traj = traj_dist.mean()

        depth_key = next((k for k in batch["observations"] if "depth" in k.lower()), None)
        traj_key = next((k for k in batch["observations"] if "future_trajectory" in k.lower()), None)
        human_num_key = next((k for k in batch["observations"] if "human_num" in k.lower()), None)

        vis_dir = os.path.join(
            self.config.habitat_baselines.checkpoint_folder, "wm_vis"
        )
        os.makedirs(vis_dir, exist_ok=True)

        mid_t = T // 2
        for i in range(num_samples):
            depth_frame = None
            if depth_key is not None:
                gt_d = batch["observations"][depth_key][i, mid_t].cpu().numpy()
                pd_d = pred_depth[i, mid_t].cpu().numpy()
                if gt_d.max() > 1.0:
                    gt_d = gt_d / 255.0
                if gt_d.ndim == 3:
                    gt_d = gt_d[..., 0]
                if pd_d.ndim == 3:
                    pd_d = pd_d[..., 0]
                if gt_d.shape != pd_d.shape:
                    pd_d = cv2.resize(pd_d, (gt_d.shape[1], gt_d.shape[0]))
                depth_frame = render_depth_comparison(gt_d, pd_d)

            traj_frame = None
            if traj_key is not None:
                gt_tr = batch["observations"][traj_key][i, mid_t].cpu().numpy()
                pd_tr = pred_traj[i, mid_t].cpu().numpy()
                if human_num_key is not None:
                    nh = int(batch["observations"][human_num_key][i, mid_t].item())
                else:
                    nh = gt_tr.shape[0]
                hp = None
                if human_state_goal is not None:
                    hp = human_state_goal[i, mid_t].cpu().numpy()[:, :2]

                traj_frame = render_trajectory_comparison(gt_tr, pd_tr, nh, human_positions=hp)

            frame = compose_wm_frame(depth_frame, traj_frame)
            if frame is not None:
                save_path = os.path.join(vis_dir, f"step{step:08d}_sample{i}.png")
                cv2.imwrite(save_path, frame)

                if writer is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                    writer.add_image(f"wm_vis/sample_{i}", frame_tensor, step)

        logger.info(f"[WM Vis] Saved {num_samples} visualization(s) to {vis_dir}")
        self._last_wm_vis_batch = None

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.sample_action"), inference_mode():
            # Sample actions
            step_batch = self._agent.rollouts.get_current_step(
                env_slice, buffer_index
            )

            profiling_wrapper.range_push("compute actions")

            # Obtain lenghts
            step_batch_lens = {
                k: v
                for k, v in step_batch.items()
                if k.startswith("index_len")
            }
            action_data = self._agent.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )

        profiling_wrapper.range_pop()  # compute actions

        with g_timer.avg_time("trainer.obs_insert"):
            for index_env, act in zip(
                range(env_slice.start, env_slice.stop),
                action_data.env_actions.cpu().unbind(0),
            ):
                if hasattr(self._agent, '_agents') and self._agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                    act = act.numpy()
                elif is_continuous_action_space(self._env_spec.action_space):
                    # Clipping actions to the specified limits
                    act = np.clip(
                        act.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                else:
                    act = act.item()
                self.envs.async_step_at(index_env, act)

        with g_timer.avg_time("trainer.obs_insert"):
            self._agent.rollouts.insert(
                next_recurrent_hidden_states=action_data.rnn_hidden_states,
                actions=action_data.actions,
                action_log_probs=action_data.action_log_probs,
                value_preds=action_data.values,
                wm_features=action_data.wm_features,
                buffer_index=buffer_index,
                should_inserts=action_data.should_inserts,
                action_data=action_data,
            )

    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.step_env"):
            outputs = [
                self.envs.wait_step_at(index_env)
                for index_env in range(env_slice.start, env_slice.stop)
            ]

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

        with g_timer.avg_time("trainer.update_stats"):
            observations = self.envs.post_step(observations)
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            rewards = torch.tensor(
                rewards_l,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            )
            rewards = rewards.unsqueeze(1)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.current_episode_reward.device,
            )
            done_masks = torch.logical_not(not_done_masks)

            self.current_episode_reward[env_slice] += rewards
            current_ep_reward = self.current_episode_reward[env_slice]
            self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
            self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore

            self._single_proc_infos = extract_scalars_from_infos(
                infos,
                ignore_keys=set(
                    k for k in infos[0].keys() if k not in self._rank0_keys
                ),
            )
            extracted_infos = extract_scalars_from_infos(
                infos, ignore_keys=self._rank0_keys
            )
            for k, v_k in extracted_infos.items():
                v = torch.tensor(
                    v_k,
                    dtype=torch.float,
                    device=self.current_episode_reward.device,
                ).unsqueeze(1)
                if k not in self.running_episode_stats:
                    self.running_episode_stats[k] = torch.zeros_like(
                        self.running_episode_stats["count"]
                    )
                self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

            self.current_episode_reward[env_slice].masked_fill_(
                done_masks, 0.0
            )

            # ===== DEBUG: Episode completion diagnostics =====
            # _n_done = done_masks.sum().item()
            # _resume_n = getattr(self, '_resume_update_num', 0)
            # _updates_since_resume = self.num_updates_done - _resume_n
            # if _n_done > 0 and _updates_since_resume <= 5:
            #     _spl_vals = extracted_infos.get("spl", [])
            #     _success_vals = extracted_infos.get("success", [])
            #     _dtg_vals = extracted_infos.get("distance_to_goal", [])
            #     _done_indices = done_masks.squeeze(-1).nonzero(as_tuple=True)[0].tolist()
            #     _done_spl = [x for i, x in enumerate(_spl_vals) if i < len(dones) and dones[i]]
            #     _done_success = [x for i, x in enumerate(_success_vals) if i < len(dones) and dones[i]]
            #     _done_dtg = [x for i, x in enumerate(_dtg_vals) if i < len(dones) and dones[i]]
            #     logger.info(
            #         f"[EPISODE DEBUG] update={self.num_updates_done} "
            #         f"done_count={int(_n_done)}, "
            #         f"spl: mean={sum(_done_spl)/max(len(_done_spl),1):.4f} "
            #         f"vals={[f'{x:.3f}' for x in _done_spl[:8]]}, "
            #         f"success: mean={sum(_done_success)/max(len(_done_success),1):.4f}, "
            #         f"dtg: mean={sum(_done_dtg)/max(len(_done_dtg),1):.2f} "
            #         f"vals={[f'{x:.1f}' for x in _done_dtg[:8]]}"
            #     )

        # Key Modification between the trainer and the original ppo trainer
        if self._is_static_encoder:
            self._encoder = self._agent.actor_critic.visual_encoder
            if self._encoder is None:
                self._encoder = self._agent._agents[0].actor_critic.visual_encoder
                with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                    batch_temp = {key.replace('agent_0_', ''): value for key, value in batch.items()}
                    batch[
                        'agent_0_' + PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch_temp)
            else:
                with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                    batch[
                        PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch)
        self._agent.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self._agent.rollouts.advance_rollout(buffer_index)

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    def _inject_bootstrap_wm_cached_feature(self, step_batch):
        """Inject rollout-cached WM feature for last-step value bootstrap."""
        wm_key = "wm_cached_feature"
        rollouts = self._agent.rollouts

        # Single-agent storage path.
        if hasattr(rollouts, "buffers"):
            if (
                "wm_features" in rollouts.buffers
                and rollouts.current_rollout_step_idx > 0
            ):
                step_batch["observations"][wm_key] = rollouts.buffers[
                    "wm_features"
                ][rollouts.current_rollout_step_idx - 1]
            return

        # Multi-agent storage path.
        if hasattr(rollouts, "_active_storages"):
            for agent_i, storage in enumerate(rollouts._active_storages):
                if storage is None or not hasattr(storage, "buffers"):
                    continue
                if (
                    "wm_features" in storage.buffers
                    and storage.current_rollout_step_idx > 0
                ):
                    step_batch["observations"][
                        f"agent_{agent_i}_{wm_key}"
                    ] = storage.buffers["wm_features"][
                        storage.current_rollout_step_idx - 1
                    ]

    @profiling_wrapper.RangeContext("_update_agent")
    @g_timer.avg_time("trainer.update_agent")
    def _update_agent(self):
        with inference_mode():
            step_batch = self._agent.rollouts.get_last_step()
            self._inject_bootstrap_wm_cached_feature(step_batch)
            step_batch_lens = {
                k: v
                for k, v in step_batch.items()
                if k.startswith("index_len")
            }

            next_value = self._agent.actor_critic.get_value(
                step_batch["observations"],
                step_batch.get("recurrent_hidden_states", None),
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )

        self._agent.rollouts.compute_returns(
            next_value,
            self._ppo_cfg.use_gae,
            self._ppo_cfg.gamma,
            self._ppo_cfg.tau,
        )

        self._agent.train()

        # Decoupled mode: ensure PPO update never touches WM params.
        if self.use_world_model:
            self._set_world_model_grad_enabled(False)
            self._clear_world_model_grads()

        losses = self._agent.updater.update(self._agent.rollouts)

        self._agent.rollouts.after_update()
        self._agent.after_update()

        if self.use_world_model:
            self._clear_world_model_grads()

        return losses

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats_before_reduce = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats_before_reduce)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i].clone())

        # ===== DEBUG: Coalesce diagnostics (commented out) =====
        # _resume_n = getattr(self, '_resume_update_num', 0)
        # _updates_since_resume = self.num_updates_done - _resume_n
        # _debug_should_log = (
        #     _updates_since_resume <= 5
        #     or self.num_updates_done % 10 == 0
        # )
        # if _debug_should_log:
        #     _debug_keys = ["count", "spl", "success", "reward", "distance_to_goal"]
        #     if self._is_distributed:
        #         _world_size = torch.distributed.get_world_size()
        #     else:
        #         _world_size = 1
        #     logger.info(
        #         f"[COALESCE DEBUG] update={self.num_updates_done} "
        #         f"(+{_updates_since_resume} since resume) "
        #         f"world_size={_world_size}, count_steps_delta={count_steps_delta}"
        #     )
        #     for k in _debug_keys:
        #         _idx = stats_ordering.index(k) if k in stats_ordering else -1
        #         if _idx < 0:
        #             continue
        #         _before = stats_before_reduce[_idx].sum().item()
        #         _after = stats[_idx].sum().item()
        #         v = self.window_episode_stats[k]
        #         _prev_sum = v[-2].sum().item() if len(v) > 1 else 0
        #         _this_increment = _after - _prev_sum
        #         logger.info(
        #             f"  {k}: local_sum={_before:.4f}, "
        #             f"all_reduce_sum={_after:.4f}, "
        #             f"prev_snapshot={_prev_sum:.4f}, "
        #             f"increment={_this_increment:.4f}, "
        #             f"window_len={len(v)}"
        #         )

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    @rank0_only
    def _training_log(
        self,
        writer,
        losses: Dict[str, float],
        prev_time: int = 0,
        count_checkpoints: Optional[int] = None,
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        # ===== FIX: Resume 后平滑过渡 =====
        # resume 后窗口从小开始增长，早期完成的 episode 以失败居多，
        # 导致指标比真实策略水平低。用旧窗口平均值做加权混合来平滑。
        # 权重 alpha = 当前窗口大小 / 目标窗口大小，alpha 从 0 → 1。
        # smoothed = alpha * raw + (1 - alpha) * old_avg
        _old_avg = getattr(self, '_resume_old_window_avg', {})
        _target_wsize = self._ppo_cfg.reward_window_size
        _cur_wsize = len(self.window_episode_stats.get("count", []))
        if _old_avg and _cur_wsize < _target_wsize:
            _alpha = _cur_wsize / _target_wsize
            for k in metrics:
                if k in _old_avg:
                    metrics[k] = _alpha * metrics[k] + (1 - _alpha) * _old_avg[k]
            if "reward" in _old_avg:
                _raw_reward = deltas["reward"] / deltas["count"]
                _smoothed_reward = _alpha * _raw_reward + (1 - _alpha) * _old_avg["reward"]
            else:
                _smoothed_reward = deltas["reward"] / deltas["count"]
        else:
            _smoothed_reward = deltas["reward"] / deltas["count"]

        writer.add_scalar(
            "reward",
            _smoothed_reward,
            self.num_steps_done,
        )

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"learner/{k}", v, self.num_steps_done)

        for k, v in self._single_proc_infos.items():
            writer.add_scalar(k, np.mean(v), self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)

        # Log perf metrics.
        writer.add_scalar("perf/fps", fps, self.num_steps_done)

        for timer_name, timer_val in g_timer.items():
            writer.add_scalar(
                f"perf/{timer_name}",
                timer_val.mean,
                self.num_steps_done,
            )

        # log stats
        if (
            self.num_updates_done % self.config.habitat_baselines.log_interval
            == 0
        ):
            logger.info("")

            # Calculate progress and timing
            progress_pct = self.percent_done() * 100
            total_steps = self.config.habitat_baselines.total_num_steps
            remaining_steps = max(total_steps - self.num_steps_done, 0)
            elapsed_time = (time.time() - self.t_start) + prev_time

            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                f"Num updates: {self.num_updates_done}\tNum frames: {self.num_steps_done}"
            )

            logger.info(
                f"Progress: {progress_pct:.2f}% ({self.num_steps_done}/{total_steps} steps) | "
                f"Remaining: {remaining_steps} steps"
            )

            if self.num_steps_done > 0:
                time_per_step = elapsed_time / self.num_steps_done
                estimated_remaining_time = time_per_step * remaining_steps
                hours_remaining = estimated_remaining_time / 3600
                logger.info(
                    f"Elapsed: {elapsed_time/3600:.2f}h | Estimated remaining: "
                    f"{hours_remaining:.2f}h ({hours_remaining/24:.2f} days)"
                )

            # Calculate next checkpoint progress
            if self.config.habitat_baselines.num_checkpoints != -1:
                num_ckpts = self.config.habitat_baselines.num_checkpoints
                checkpoint_interval_steps = total_steps / num_ckpts
                next_segment_1based = int(
                    self.num_steps_done / checkpoint_interval_steps
                ) + 1
                next_checkpoint_steps = (
                    next_segment_1based * checkpoint_interval_steps
                )
                steps_to_next_ckpt = max(
                    int(next_checkpoint_steps - self.num_steps_done), 0
                )
                pct_to_next_ckpt = (
                    (self.num_steps_done % checkpoint_interval_steps)
                    / checkpoint_interval_steps
                    * 100
                )
                # 显示实际将要保存的文件名 (count_checkpoints)，与 "Saved ckpt.X.pth" 一致
                next_save_str = (
                    f"Next save: ckpt.{count_checkpoints}.pth"
                    if count_checkpoints is not None
                    else f"Next segment: {next_segment_1based}/{num_ckpts}"
                )
                logger.info(
                    f"{next_save_str} | "
                    f"Progress to next: {pct_to_next_ckpt:.1f}% "
                    f"({steps_to_next_ckpt} steps remaining)"
                )

            if self.train_world_model:
                logger.info(
                    "WM replay status: "
                    f"{self._get_replay_buffer_size()}/{self.replay_buffer_size} "
                    f"(warmup={self.replay_buffer_warmup})"
                )

            logger.info("  --- Losses ---")
            for k, v in sorted(losses.items()):
                logger.info(f"    {k}: {v:.4f}")

            _is_smoothing = bool(_old_avg and _cur_wsize < _target_wsize)
            if _is_smoothing:
                logger.info(
                    "  --- 窗口指标 (window_size={}, smoothing alpha={:.2f}) ---".format(
                        _cur_wsize, _alpha
                    )
                )
            else:
                logger.info(
                    "  --- 窗口指标 (window_size={}) ---".format(
                        len(self.window_episode_stats["count"])
                    )
                )
            logger.info(f"    count: {deltas['count']:.4f}")
            for k in sorted(metrics.keys()):
                _raw = deltas[k] / deltas["count"] if k in deltas else 0
                if _is_smoothing and k in _old_avg:
                    logger.info(f"    {k}: {metrics[k]:.4f} (raw={_raw:.4f}, old={_old_avg[k]:.4f})")
                else:
                    logger.info(f"    {k}: {metrics[k]:.4f}")

            # ===== DEBUG: Window detail for key metrics (commented out) =====
            # _debug_detail_keys = ["count", "spl", "success", "distance_to_goal", "reward"]
            # _resume_n = getattr(self, '_resume_update_num', 0)
            # _updates_since_resume = self.num_updates_done - _resume_n
            # logger.info(f"  --- [DEBUG] 窗口详情 (update={self.num_updates_done}, "
            #             f"+{_updates_since_resume} since resume) ---")
            # for k in _debug_detail_keys:
            #     if k in self.window_episode_stats:
            #         v = self.window_episode_stats[k]
            #         wlen = len(v)
            #         first_sum = v[0].sum().item() if wlen > 0 else 0
            #         last_sum = v[-1].sum().item() if wlen > 0 else 0
            #         delta = (v[-1] - v[0]).sum().item() if wlen > 1 else (v[0].sum().item() if wlen > 0 else 0)
            #         last_incr = (v[-1] - v[-2]).sum().item() if wlen > 1 else 0
            #         logger.info(
            #             f"    {k}: wlen={wlen}, "
            #             f"first={first_sum:.4f}, last={last_sum:.4f}, "
            #             f"delta={delta:.4f}, "
            #             f"last_incr={last_incr:.4f}, "
            #             f"avg={delta / max(deltas['count'], 1.0):.4f}"
            #         )
            #         if wlen <= 6:
            #             all_sums = [vi.sum().item() for vi in v]
            #             logger.info(f"      all_snapshots: {[f'{s:.2f}' for s in all_sums]}")
            #         elif wlen <= 10:
            #             all_sums = [vi.sum().item() for vi in v]
            #             logger.info(f"      snapshots: {[f'{s:.1f}' for s in all_sums]}")
            #         else:
            #             head_sums = [v[j].sum().item() for j in range(3)]
            #             tail_sums = [v[j].sum().item() for j in range(wlen-3, wlen)]
            #             logger.info(
            #                 f"      first_3: {[f'{s:.2f}' for s in head_sums]}, "
            #                 f"last_3: {[f'{s:.2f}' for s in tail_sums]}"
            #             )

            logger.info("  --- Perf (mean) ---")
            for k, v in g_timer.items():
                logger.info(f"    {k}: {v.mean:.3f}s")

            if self.config.habitat_baselines.should_log_single_proc_infos:
                logger.info("  --- Single-proc infos ---")
                for k, v in self._single_proc_infos.items():
                    logger.info(f"    {k}: {np.mean(v):.4f}")

            if torch.cuda.is_available():
                try:
                    alloc = torch.cuda.memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    logger.info(
                        "  --- GPU ---  "
                        f"alloc: {alloc:.2f} GB  |  reserved: {reserved:.2f} GB"
                    )
                except Exception:
                    pass

            logger.info("=" * 70)

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step
            >= self.config.habitat_baselines.rl.ppo.num_steps
            * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.habitat_baselines.rl.ddppo.sync_frac
            * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        resume_state = load_resume_state(self.config)
        self._init_train(resume_state)

        count_checkpoints = 0
        prev_time = 0

        if self._is_distributed:
            torch.distributed.barrier()

        resume_run_id = None
        if resume_state is not None:
            self._agent.load_state_dict(resume_state)

            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            # ===== FIX: Resume 后窗口指标连续性修复 =====
            # 不恢复 running_episode_stats 和 window_episode_stats。
            # 原因：resume 后环境被完全重建，所有 episode 从头开始，
            # running_episode_stats 的旧累计值与新环境状态不匹配。
            # 让两者都保持 _init_train 中初始化的零值，与从头训练一致。
            # 这样窗口指标忠实反映 resume 后的真实策略表现。
            #
            # 注意：不恢复 running_episode_stats 意味着旧累计数据丢失，
            # 但窗口指标只关心增量（v[-1] - v[0]），不依赖历史绝对值。
            # logger.info("[RESUME] NOT restoring running_episode_stats and "
            #             "window_episode_stats (reset to zeros, consistent with "
            #             "newly rebuilt environments)")

            # 保存旧窗口平均值用于平滑过渡
            _old_window = requeue_stats.get("window_episode_stats", {})
            _old_window_len = len(next(iter(_old_window.values()), []))
            self._resume_old_window_avg = {}
            if _old_window_len > 1 and "count" in _old_window:
                _old_count_delta = (_old_window['count'][-1] - _old_window['count'][0]).sum().item()
                if _old_count_delta > 0:
                    for _ok, _ov in _old_window.items():
                        if _ok == "count":
                            continue
                        _ov_delta = (_ov[-1] - _ov[0]).sum().item()
                        self._resume_old_window_avg[_ok] = _ov_delta / _old_count_delta
                # logger.info(f"[RESUME] Old window averages (for reference, {_old_window_len} snapshots):")
                # for _ok in ["count", "spl", "success", "reward", "distance_to_goal"]:
                #     if _ok in self._resume_old_window_avg:
                #         logger.info(f"  {_ok}: {self._resume_old_window_avg[_ok]:.4f}")
                #     elif _ok == "count":
                #         logger.info(f"  count: delta={_old_count_delta:.0f}")

            resume_run_id = requeue_stats.get("run_id", None)

            self._resume_update_num = self.num_updates_done

            # ===== DEBUG: Resume diagnostics (commented out) =====
            # logger.info("=" * 70)
            # logger.info(f"[RESUME DEBUG] num_updates_done={self.num_updates_done}, "
            #             f"num_steps_done={self.num_steps_done}, "
            #             f"num_envs={self.envs.num_envs}, "
            #             f"is_distributed={self._is_distributed}")
            # logger.info("[RESUME DEBUG] running_episode_stats: reset to zeros "
            #             f"(num_envs={self.envs.num_envs})")
            # logger.info("[RESUME DEBUG] window_episode_stats: empty")
            # logger.info("=" * 70)

        with (
            get_writer(
                self.config,
                resume_run_id=resume_run_id,
                flush_secs=self.flush_secs,
                purge_step=int(self.num_steps_done),
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                self._agent.pre_rollout()

                # Resume state on preemption signal only (resume 与 ckpt 同频，见下方 save_checkpoint 处)
                if rank0_only() and SAVE_STATE.is_set():
                    requeue_stats = dict(
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                        run_id=writer.get_run_id(),
                    )
                    save_resume_state(
                        dict(
                            **self._agent.get_resume_state(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self._agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                with g_timer.avg_time("trainer.rollout_collect"):
                    for buffer_index in range(self._agent.nbuffers):
                        self._compute_actions_and_step_envs(buffer_index)

                    for step in range(self._ppo_cfg.num_steps):
                        is_last_step = (
                            self.should_end_early(step + 1)
                            or (step + 1) == self._ppo_cfg.num_steps
                        )

                        for buffer_index in range(self._agent.nbuffers):
                            count_steps_delta += (
                                self._collect_environment_result(buffer_index)
                            )

                            if (buffer_index + 1) == self._agent.nbuffers:
                                profiling_wrapper.range_pop()  # _collect_rollout_step

                            if not is_last_step:
                                if (buffer_index + 1) == self._agent.nbuffers:
                                    profiling_wrapper.range_push(
                                        "_collect_rollout_step"
                                    )

                                self._compute_actions_and_step_envs(
                                    buffer_index
                                )

                        if is_last_step:
                            break

                profiling_wrapper.range_pop()  # rollouts loop

                # ==================== Store rollouts to replay buffer ====================
                if self.train_world_model:
                    self._store_rollout_to_buffer(self._agent.rollouts)

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                # ==================== Policy Update ====================
                losses = self._update_agent()

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    losses,
                    count_steps_delta,
                )

                # ==================== World Model Update (if needed) ====================
                if self._should_update_world_model(self.num_updates_done):
                    wm_losses = self._update_world_model(self.num_updates_done)
                    # 添加 WM losses 到总 losses
                    for key, value in wm_losses.items():
                        losses[f'wm_{key}'] = value
                    # 每 10000 次 update 才存储 WM 可视化到磁盘并写 TensorBoard
                    if rank0_only() and (self.num_updates_done % 10000 == 0):
                        self._visualize_wm_predictions(
                            writer, self.num_steps_done, num_samples=3
                        )

                self._training_log(
                    writer, losses, prev_time,
                    count_checkpoints=count_checkpoints,
                )

                # checkpoint model（resume state 与 ckpt 同频保存）
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    logger.info(f"Saved checkpoint ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1
                    # 与 ckpt 同频写 resume state，便于断点续训
                    requeue_stats = dict(
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                        run_id=writer.get_run_id(),
                    )
                    save_resume_state(
                        dict(
                            **self._agent.get_resume_state(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
            step_id = ckpt_dict["extra_state"]["step"]
            logger.info(f"Loaded checkpoint trained for {step_id} steps")
        else:
            ckpt_dict = {"config": None}

        if "config" not in ckpt_dict:
            ckpt_dict["config"] = None

        config = self._get_resume_state_config_or_new_config(
            ckpt_dict["config"]
        )
        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            n_agents = len(config.habitat.simulator.agents)
            for agent_i in range(n_agents):
                agent_name = config.habitat.simulator.agents_order[agent_i]
                agent_config = get_agent_config(
                    config.habitat.simulator, agent_i
                )

                agent_sensors = agent_config.sim_sensors
                extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
                with read_write(agent_sensors):
                    agent_sensors.update(extra_sensors)
                with read_write(config):
                    if config.habitat.gym.obs_keys is not None:
                        for render_view in extra_sensors.values():
                            if (
                                render_view.uuid
                                not in config.habitat.gym.obs_keys
                            ):
                                if n_agents > 1:
                                    config.habitat.gym.obs_keys.append(
                                        f"{agent_name}_{render_view.uuid}"
                                    )
                                else:
                                    config.habitat.gym.obs_keys.append(
                                        render_view.uuid
                                    )

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        self._agent = self._create_agent(None)
        if (
            self._agent.actor_critic.should_load_agent_state
            and self.config.habitat_baselines.eval.should_load_ckpt
        ):
            self._agent.load_state_dict(ckpt_dict)

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        evaluator = hydra.utils.instantiate(config.habitat_baselines.evaluator)
        assert isinstance(evaluator, Evaluator)
        evaluator.evaluate_agent(
            self._agent,
            self.envs,
            self.config,
            checkpoint_index,
            step_id,
            writer,
            self.device,
            self.obs_transforms,
            self._env_spec,
            self._rank0_keys,
        )

        self.envs.close()


def get_device(config: "DictConfig") -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda", config.habitat_baselines.torch_gpu_id)
        torch.cuda.set_device(device)
        return device
    else:
        return torch.device("cpu")
