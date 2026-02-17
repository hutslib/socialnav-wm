# Copyright (c) 2024 ForeSightNav-WM
# Social Navigation World Model Implementation (V2)
#
# This is the V2 version designed for Falcon policy enhancement.
# Key differences from V1:
# - No standalone train_step() - training handled by falcon_trainer.py
# - No imagine() - V2 doesn't use imagination rollouts
# - No internal optimizer - optimizer created externally in trainer
# - Only provides core components: encoder, dynamics (RSSM), and heads
#
# Encoder options:
# - 'dreamer': Dreamer-style CNN (default, designed for WM)
# - 'falcon': Reuse Falcon's ResNet encoder (can use pretrained weights)

import numpy as np
import torch
from torch import nn
from habitat_baselines.rl.world_model.networks import (
    RSSM,
    SocialNavEncoder,
    DepthDecoder,
    HumanTrajectoryDecoder,
    RewardDecoder,
)
from habitat_baselines.rl.world_model.falcon_encoder_adapter import FalconEncoderAdapter
from habitat_baselines.rl.ddppo.policy.resnet import resnet18, resnet50


class SocialNavWorldModel(nn.Module):
    """
    Human-Forecasting Latent World Model for Social Navigation (V2)
    
    Provides core World Model components for Falcon policy enhancement:
    - Encoder: Observations → Embeddings
    - RSSM Dynamics: Latent state transitions
    - Decoders: Depth reconstruction, Human trajectory prediction, Reward prediction
    
    Training is managed externally by falcon_trainer.py which directly
    calls these components for maximum flexibility.
    """
    def __init__(self, config, observation_space=None, device="cuda"):
        super(SocialNavWorldModel, self).__init__()
        self._config = config
        self.device = device

        # Encoder selection: 'dreamer' (default) or 'falcon'
        encoder_type = config.get('encoder_type', 'dreamer')
        
        if encoder_type == 'falcon':
            # Use Falcon's ResNet encoder with adapter
            print("Using Falcon ResNet encoder (can reuse pretrained weights)")
            
            if observation_space is None:
                raise ValueError("observation_space is required for Falcon encoder")
            
            # Select ResNet backbone
            backbone_name = config.get('falcon_backbone', 'resnet18')
            make_backbone = resnet18 if backbone_name == 'resnet18' else resnet50

            self.encoder = FalconEncoderAdapter(
                observation_space=observation_space,
                baseplanes=config.get('falcon_baseplanes', 32),
                ngroups=config.get('falcon_ngroups', 32),
                make_backbone=make_backbone,
                hidden_size=config.get('mlp_units', 256),
                use_projection=config.get('falcon_use_projection', False),
                target_dim=config.get('falcon_target_dim', 512),
                visual_keys=config.get('visual_keys'),
                fuse_keys_1d=config.get('fuse_keys_1d'),
                human_state_goal_keys=config.get('human_state_goal_keys'),
            )

            # Load pretrained Falcon weights if specified
            if config.get('falcon_pretrained_path', None):
                self.encoder.load_falcon_weights(
                    config['falcon_pretrained_path'],
                    strict=config.get('falcon_strict_load', False)
                )
        
        elif encoder_type == 'dreamer':
            # Use Dreamer-style CNN encoder (default)
            print("Using Dreamer-style CNN encoder")
            
            if observation_space is not None:
                self.encoder = SocialNavEncoder(
                    observation_space=observation_space,
                    depth=config.get('cnn_depth', 32),
                    act=config.get('act', 'SiLU'),
                    norm=config.get('norm', True),
                    kernel_size=config.get('kernel_size', 4),
                    minres=config.get('minres', 4),
                    hidden_size=config.get('mlp_units', 256),
                    visual_keys=config.get('visual_keys'),
                    fuse_keys_1d=config.get('fuse_keys_1d'),
                    human_state_goal_keys=config.get('human_state_goal_keys'),
                )
            else:
                # Fallback to old interface for compatibility
                from gym import spaces
                dummy_obs_space = spaces.Dict({
                    'depth': spaces.Box(
                        low=0, high=255,
                        shape=config.get('depth_shape', (256, 256, 1)),
                        dtype=np.uint8
                    )
                })
                self.encoder = SocialNavEncoder(
                    observation_space=dummy_obs_space,
                    depth=config.get('cnn_depth', 32),
                    act=config.get('act', 'SiLU'),
                    norm=config.get('norm', True),
                    kernel_size=config.get('kernel_size', 4),
                    minres=config.get('minres', 4),
                    hidden_size=config.get('mlp_units', 256),
                )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'dreamer' or 'falcon'")
        
        self.embed_size = self.encoder.outdim

        # RSSM dynamics model
        self.dynamics = RSSM(
            stoch=config.get('dyn_stoch', 30),
            deter=config.get('dyn_deter', 200),
            hidden=config.get('dyn_hidden', 200),
            rec_depth=config.get('dyn_rec_depth', 1),
            discrete=config.get('dyn_discrete', 32),
            act=config.get('act', 'SiLU'),
            norm=config.get('norm', True),
            mean_act=config.get('dyn_mean_act', 'none'),
            std_act=config.get('dyn_std_act', 'softplus'),
            min_std=config.get('dyn_min_std', 0.1),
            unimix_ratio=config.get('unimix_ratio', 0.01),
            initial=config.get('initial', 'learned'),
            num_actions=config.get('num_actions', 4),
            embed_size=self.embed_size,
            device=device,
        )

        # Feature size (stoch + deter)
        if config.get('dyn_discrete', 32):
            feat_size = config.get('dyn_stoch', 30) * config.get('dyn_discrete', 32) + \
                       config.get('dyn_deter', 200)
        else:
            feat_size = config.get('dyn_stoch', 30) + config.get('dyn_deter', 200)

        # Decoders
        self.heads = nn.ModuleDict()
        
        # Depth reconstruction decoder
        self.heads["depth"] = DepthDecoder(
            feat_size=feat_size,
            depth_shape=config.get('depth_shape', (256, 256, 1)),
            depth=config.get('decoder_depth', 32),
            act=config.get('act', 'SiLU'),
            norm=config.get('norm', True),
            kernel_size=config.get('kernel_size', 4),
            minres=config.get('minres', 4),
            outscale=config.get('decoder_outscale', 1.0),
        )
        
        # Human trajectory prediction decoder（goal conditioning 使用 human_state_goal）
        self.heads["human_traj"] = HumanTrajectoryDecoder(
            feat_size=feat_size,
            num_humans=config.get('num_humans', 10),
            pred_horizon=config.get('pred_horizon', 12),
            traj_dim=config.get('traj_dim', 2),
            hidden_layers=config.get('traj_hidden_layers', 3),
            hidden_units=config.get('traj_hidden_units', 256),
            act=config.get('act', 'SiLU'),
            norm=config.get('norm', True),
            outscale=config.get('decoder_outscale', 1.0),
            use_goal_conditioning=config.get('use_goal_conditioning', True),
            state_goal_dim=config.get('state_goal_dim', 8),
        )
        
        # Reward prediction decoder
        self.heads["reward"] = RewardDecoder(
            feat_size=feat_size,
            hidden_size=config.get('reward_hidden', 256),
            act=config.get('act', 'SiLU'),
            norm=config.get('norm', True),
            outscale=config.get('decoder_outscale', 1.0),
        )

    def get_feat(self, state):
        """
        Get feature vector from RSSM state.
        
        Used by both policy (falcon_policy_v2.py) and trainer (falcon_trainer.py).
        
        Args:
            state: RSSM state dict with 'stoch' and 'deter'
        
        Returns:
            Feature vector [batch, feat_size]
        """
        return self.dynamics.get_feat(state)
