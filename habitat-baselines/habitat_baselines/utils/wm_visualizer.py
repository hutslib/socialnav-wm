"""
World Model visualization utilities for evaluation.

Generates side-by-side comparison frames for:
  - Depth: ground-truth vs. WM-reconstructed depth maps
  - Trajectory: ground-truth vs. WM-predicted human trajectories

These frames are collected per step and assembled into videos
alongside the standard eval video.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


# GT: dark saturated (BGR) — easy to see with transparency
_HUMAN_COLORS_GT = [
    (0, 50, 200),    # dark red
    (80, 120, 0),    # dark green
    (200, 80, 0),    # dark blue
    (0, 140, 200),   # dark orange
    (200, 0, 150),   # dark magenta
    (180, 180, 0),   # dark cyan
    (100, 0, 200),
    (0, 180, 180),
    (180, 0, 100),
    (100, 150, 0),
]

# Pred: bright, different hue from GT (BGR) — opaque
_HUMAN_COLORS_PRED = [
    (0, 165, 255),   # orange
    (130, 255, 130), # light green
    (255, 130, 0),   # bright blue
    (0, 215, 255),   # gold/orange
    (255, 80, 200),  # magenta
    (255, 255, 80),  # cyan
    (255, 0, 255),   # magenta
    (180, 255, 180),
    (200, 80, 255),
    (100, 255, 150),
]


def depth_to_colormap(depth, vmin=0.0, vmax=1.0):
    depth_clipped = np.clip(depth, vmin, vmax)
    if vmax - vmin > 1e-6:
        depth_norm = ((depth_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth_clipped, dtype=np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)


def render_depth_comparison(gt_depth, pred_depth, target_h=256, target_w=256):
    if gt_depth.ndim == 3:
        gt_depth = gt_depth[..., 0]
    if pred_depth.ndim == 3:
        pred_depth = pred_depth[..., 0]

    gt_vis = depth_to_colormap(gt_depth)
    pred_vis = depth_to_colormap(pred_depth)

    gt_vis = cv2.resize(gt_vis, (target_w, target_h))
    pred_vis = cv2.resize(pred_vis, (target_w, target_h))

    gap = 4
    canvas = np.ones((target_h, 2 * target_w + gap, 3), dtype=np.uint8) * 255
    canvas[:, :target_w] = gt_vis
    canvas[:, target_w + gap:] = pred_vis

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "GT Depth", (5, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "WM Pred", (target_w + gap + 5, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    rmse = float(np.sqrt(np.mean((gt_depth - pred_depth) ** 2)))
    cv2.putText(canvas, f"RMSE={rmse:.4f}", (target_w + gap + 5, target_h - 10),
                font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def _alpha_overlay(base, overlay, alpha=0.55):
    """Blend *overlay* onto *base* with constant alpha (in-place on base)."""
    mask = np.any(overlay != 0, axis=-1)
    base[mask] = (
        base[mask].astype(np.float32) * (1 - alpha)
        + overlay[mask].astype(np.float32) * alpha
    ).astype(np.uint8)


def render_trajectory_comparison(
    gt_traj, pred_traj, num_humans,
    human_positions=None,
    canvas_size=512, world_range=8.0,
):
    """
    Args:
        gt_traj: (N, T, 2) ground-truth future trajectory (robot-centric).
        pred_traj: (N, T, 2) WM-predicted future trajectory.
        num_humans: number of valid humans.
        human_positions: optional (N, 2) current positions from human_state_goal
                         pos_x, pos_z (robot-centric). Drawn as cross markers.
    """
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

    cx, cy = canvas_size // 2, canvas_size // 2
    scale = canvas_size / (2 * world_range)

    def world_to_px(pos):
        px = int(cx + pos[0] * scale)
        py = int(cy + pos[1] * scale)
        return (np.clip(px, 0, canvas_size - 1), np.clip(py, 0, canvas_size - 1))

    cv2.drawMarker(canvas, (cx, cy), (80, 80, 80), cv2.MARKER_DIAMOND, 14, 2)

    gt_layer = np.zeros_like(canvas)

    for h in range(min(num_humans, len(gt_traj), len(pred_traj))):
        gt_h = gt_traj[h]
        pred_h = pred_traj[h]

        if np.all(np.abs(gt_h) > 90):
            continue

        gt_color = _HUMAN_COLORS_GT[h % len(_HUMAN_COLORS_GT)]
        pd_color = _HUMAN_COLORS_PRED[h % len(_HUMAN_COLORS_PRED)]

        hp = None
        if human_positions is not None and h < len(human_positions):
            _hp = human_positions[h]
            if not np.all(np.abs(_hp) > 90):
                hp = _hp

        # Predicted trajectory — solid line; prepend segment CurPos -> pred[0] so line starts at X
        if len(pred_h) > 0:
            if hp is not None:
                cv2.line(canvas, world_to_px(hp), world_to_px(pred_h[0]), pd_color, 3, cv2.LINE_AA)
            for t in range(len(pred_h) - 1):
                cv2.line(canvas, world_to_px(pred_h[t]), world_to_px(pred_h[t + 1]), pd_color, 3, cv2.LINE_AA)
            cv2.circle(canvas, world_to_px(pred_h[0]), 6, pd_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, world_to_px(pred_h[-1]), 5, pd_color, 2, cv2.LINE_AA)

        # GT trajectory — solid line + waypoint marker at every path point
        if len(gt_h) > 0:
            if hp is not None:
                cv2.line(gt_layer, world_to_px(hp), world_to_px(gt_h[0]), gt_color, 3, cv2.LINE_AA)
            for t in range(len(gt_h) - 1):
                cv2.line(gt_layer, world_to_px(gt_h[t]), world_to_px(gt_h[t + 1]), gt_color, 3, cv2.LINE_AA)
            for t in range(len(gt_h)):
                pt = world_to_px(gt_h[t])
                if t == 0:
                    cv2.circle(gt_layer, pt, 6, gt_color, -1, cv2.LINE_AA)
                elif t == len(gt_h) - 1:
                    cv2.circle(gt_layer, pt, 5, gt_color, -1, cv2.LINE_AA)
                    cv2.circle(gt_layer, pt, 5, gt_color, 2, cv2.LINE_AA)
                else:
                    cv2.circle(gt_layer, pt, 4, gt_color, -1, cv2.LINE_AA)

    _alpha_overlay(canvas, gt_layer, alpha=0.6)

    # Draw current position (X) on top so it is not covered by trajectories
    for h in range(min(num_humans, len(gt_traj), len(pred_traj))):
        if np.all(np.abs(gt_traj[h]) > 90):
            continue
        if human_positions is not None and h < len(human_positions):
            hp = human_positions[h]
            if not np.all(np.abs(hp) > 90):
                color = _HUMAN_COLORS_GT[h % len(_HUMAN_COLORS_GT)]
                cv2.drawMarker(canvas, world_to_px(hp), color,
                               cv2.MARKER_CROSS, 14, 3, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (60, 60, 60)
    cv2.putText(canvas, "GT=transparent  Pred=opaque  X=CurPos", (5, 20),
                font, 0.42, text_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Humans: {num_humans}", (5, canvas_size - 10),
                font, 0.5, text_color, 1, cv2.LINE_AA)

    if num_humans > 0:
        valid = min(num_humans, len(gt_traj), len(pred_traj))
        ade_vals = []
        for h in range(valid):
            if np.all(np.abs(gt_traj[h]) > 90):
                continue
            disp = np.linalg.norm(gt_traj[h] - pred_traj[h], axis=-1)
            ade_vals.append(disp.mean())
        if ade_vals:
            ade = float(np.mean(ade_vals))
            cv2.putText(canvas, f"ADE={ade:.3f}m", (canvas_size - 160, canvas_size - 10),
                        font, 0.5, text_color, 1, cv2.LINE_AA)

    return canvas


def compose_wm_frame(depth_frame, traj_frame, target_h=256):
    panels = []
    if depth_frame is not None:
        h_ratio = target_h / depth_frame.shape[0]
        new_w = int(depth_frame.shape[1] * h_ratio)
        panels.append(cv2.resize(depth_frame, (new_w, target_h)))
    if traj_frame is not None:
        h_ratio = target_h / traj_frame.shape[0]
        new_w = int(traj_frame.shape[1] * h_ratio)
        panels.append(cv2.resize(traj_frame, (new_w, target_h)))

    if not panels:
        return None

    gap = 4
    total_w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.ones((target_h, total_w, 3), dtype=np.uint8) * 255
    x = 0
    for p in panels:
        canvas[:, x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    return canvas


class WMVisualizer:
    """
    Stateful helper that runs WM inference each eval step and
    produces visualisation frames.
    """

    def __init__(self, world_model, device, num_envs=1):
        self.world_model = world_model
        self.device = device
        self.num_envs = num_envs
        self.rssm_states = [None] * num_envs
        self._step_count = 0

    def reset_env(self, env_idx):
        self.rssm_states[env_idx] = None

    @torch.no_grad()
    def step(self, batch, prev_actions, masks, env_idx):
        wm = self.world_model
        if wm is None:
            return None

        _log = self._step_count < 3
        self._step_count += 1

        obs_i = {}
        for k, v in batch.items():
            val = v[env_idx:env_idx + 1]
            obs_i[k] = val
            stripped = k.replace("agent_0_", "", 1) if k.startswith("agent_0_") else None
            if stripped is not None:
                obs_i[stripped] = val

        if _log:
            print(f"[WMVis] obs keys: {list(obs_i.keys())}")
            for k, v in obs_i.items():
                print(f"[WMVis]   {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

        embed = wm.encoder(obs_i)
        if _log:
            print(f"[WMVis] embed: {tuple(embed.shape)}")

        if self.rssm_states[env_idx] is None or not masks[env_idx].any().item():
            self.rssm_states[env_idx] = wm.dynamics.initial(1)

        action_i = prev_actions[env_idx:env_idx + 1]
        if action_i.dim() > 1:
            action_i = action_i.squeeze(-1)
        num_actions = wm.dynamics._num_actions
        if action_i.dtype in (torch.long, torch.int32, torch.int64):
            action_oh = F.one_hot(action_i.clamp(min=0), num_classes=num_actions).float()
        else:
            action_oh = action_i.unsqueeze(0) if action_i.dim() == 0 else action_i

        embed_seq = embed.unsqueeze(1)
        action_seq = action_oh.unsqueeze(1)
        is_first = (~masks[env_idx:env_idx + 1]).float()

        post, _ = wm.dynamics.observe(
            embed_seq, action_seq, is_first, self.rssm_states[env_idx]
        )
        self.rssm_states[env_idx] = {k: v[:, 0] for k, v in post.items()}

        feat = wm.dynamics.get_feat(post)
        if _log:
            print(f"[WMVis] feat: {tuple(feat.shape)}")

        # Depth reconstruction
        depth_frame = None
        depth_key = next(
            (k for k in obs_i if "depth" in k.lower()), None
        )
        if depth_key is not None:
            skip_feats = getattr(wm.encoder, '_cached_visual_feats_ms', None)
            depth_dist = wm.heads["depth"](feat, skip_feats=skip_feats)
            pred_depth = depth_dist.mean()
            if _log:
                print(f"[WMVis] gt_depth: {tuple(obs_i[depth_key][0].shape)}, pred_depth: {tuple(pred_depth.shape)}")

            gt_depth_raw = obs_i[depth_key][0].cpu().numpy()
            if gt_depth_raw.max() > 1.0:
                gt_depth_raw = gt_depth_raw / 255.0
            if gt_depth_raw.ndim == 3:
                gt_depth_np = gt_depth_raw[..., 0]
            else:
                gt_depth_np = gt_depth_raw

            pred_depth_np = pred_depth[0, 0].cpu().numpy()
            if pred_depth_np.ndim == 3:
                pred_depth_np = pred_depth_np[..., 0]

            if gt_depth_np.shape != pred_depth_np.shape:
                pred_depth_np = cv2.resize(pred_depth_np, (gt_depth_np.shape[1], gt_depth_np.shape[0]))

            depth_frame = render_depth_comparison(gt_depth_np, pred_depth_np)

        # Trajectory prediction
        traj_frame = None
        traj_key = next(
            (k for k in obs_i if "future_trajectory" in k.lower()), None
        )
        if traj_key is not None:
            human_state_goal_key = next(
                (k for k in obs_i if "human_state_goal" in k), None
            )
            human_state_goal = obs_i[human_state_goal_key] if human_state_goal_key else None
            if _log:
                hsg_shape = tuple(human_state_goal.shape) if human_state_goal is not None else None
                print(f"[WMVis] feat: {tuple(feat.shape)}, human_state_goal (raw): {hsg_shape}")
            # feat is (batch, time=1, feat_dim) from observe(); human_state_goal
            # is (batch, N, 8) without a time axis — always insert time=1.
            if human_state_goal is not None:
                human_state_goal = human_state_goal.unsqueeze(1)
            if _log:
                hsg_shape = tuple(human_state_goal.shape) if human_state_goal is not None else None
                print(f"[WMVis] human_state_goal (after unsqueeze): {hsg_shape}")

            traj_dist = wm.heads["human_traj"](feat, human_state_goal=human_state_goal)
            pred_traj = traj_dist.mean()
            if _log:
                print(f"[WMVis] gt_traj: {tuple(obs_i[traj_key][0].shape)}, pred_traj: {tuple(pred_traj.shape)}")

            gt_traj_raw = obs_i[traj_key][0].cpu().numpy()
            pred_traj_np = pred_traj[0, 0].cpu().numpy()

            human_num_key = next(
                (k for k in obs_i if "human_num" in k.lower()), None
            )
            if human_num_key is not None:
                num_humans = int(obs_i[human_num_key][0].item())
            else:
                num_humans = gt_traj_raw.shape[0]

            # Extract current human positions from human_state_goal[:, 0:2]
            human_positions = None
            if human_state_goal_key is not None:
                hsg_np = obs_i[human_state_goal_key][0].cpu().numpy()  # (N, 8)
                human_positions = hsg_np[:, :2]  # pos_x, pos_z

            traj_frame = render_trajectory_comparison(
                gt_traj_raw, pred_traj_np, num_humans,
                human_positions=human_positions,
            )

        return compose_wm_frame(depth_frame, traj_frame)
