"""
World Model visualization utilities for evaluation.

Generates visualization components for:
  - Depth: WM-reconstructed depth colormap
  - Trajectory: GT vs. WM-predicted human trajectories overlaid on top-down map

These components are composed into a unified eval video frame alongside
the standard RGB observation.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


# Default GT future trajectory colors (RGB) — same hue as history trajectory in maps.py.
_DEFAULT_GT_COLORS = [
    (200, 50, 0),     # H1: dark red
    (0, 120, 80),     # H2: dark green
    (0, 80, 200),     # H3: dark blue
    (200, 140, 0),    # H4: dark orange
    (150, 0, 200),    # H5: dark magenta
    (0, 180, 180),    # H6: dark cyan
    (200, 0, 100),    # H7: red-violet
    (0, 150, 100),    # H8: teal
]

# Default Pred future trajectory colors (RGB) — brighter/lighter version of GT hue
_DEFAULT_PRED_COLORS = [
    (255, 120, 80),   # H1: bright red-orange
    (100, 255, 130),  # H2: bright green
    (60, 160, 255),   # H3: bright blue
    (255, 215, 0),    # H4: bright gold
    (220, 80, 255),   # H5: bright magenta
    (80, 255, 255),   # H6: bright cyan
    (255, 60, 180),   # H7: bright pink
    (80, 255, 150),   # H8: bright teal
]

_DEFAULT_ROBOT_GOAL_COLOR = (200, 0, 0)


def _get_traj_cfg_value(traj_cfg, key, default):
    """Read a value from traj_cfg (OmegaConf / dict / dataclass), fallback to default."""
    if traj_cfg is None:
        return default
    if hasattr(traj_cfg, key):
        return getattr(traj_cfg, key)
    if isinstance(traj_cfg, dict):
        return traj_cfg.get(key, default)
    return default


def _parse_color_list(raw, default):
    """Convert config color list (list of lists) to list of tuples."""
    if raw is None or raw is default:
        return default
    return [tuple(c) for c in raw]


@dataclass
class WMStepResult:
    """Per-step WM inference results for composing the unified eval frame."""
    pred_depth_vis: Optional[np.ndarray] = None   # (H, W, 3) colormap (legacy / trainer compat)
    pred_depth_raw: Optional[np.ndarray] = None   # (H, W) float32 raw predicted depth
    gt_traj: Optional[np.ndarray] = None           # (N, T, 2)
    pred_traj: Optional[np.ndarray] = None         # (N, T, 2)
    num_humans: int = 0
    human_positions: Optional[np.ndarray] = None   # (N, 2) current pos, robot-centric
    human_goals: Optional[np.ndarray] = None       # (N, 2) goal pos, robot-centric
    depth_rmse: float = 0.0
    traj_ade: float = 0.0


_COLORMAP_TABLE = {
    "AUTUMN": cv2.COLORMAP_AUTUMN,
    "BONE": cv2.COLORMAP_BONE,
    "JET": cv2.COLORMAP_JET,
    "WINTER": cv2.COLORMAP_WINTER,
    "RAINBOW": cv2.COLORMAP_RAINBOW,
    "OCEAN": cv2.COLORMAP_OCEAN,
    "SUMMER": cv2.COLORMAP_SUMMER,
    "SPRING": cv2.COLORMAP_SPRING,
    "COOL": cv2.COLORMAP_COOL,
    "HSV": cv2.COLORMAP_HSV,
    "PINK": cv2.COLORMAP_PINK,
    "HOT": cv2.COLORMAP_HOT,
    "PARULA": cv2.COLORMAP_PARULA,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "CIVIDIS": cv2.COLORMAP_CIVIDIS,
    "TWILIGHT": cv2.COLORMAP_TWILIGHT,
    "TWILIGHT_SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED,
    "TURBO": cv2.COLORMAP_TURBO,
    "DEEPGREEN": cv2.COLORMAP_DEEPGREEN,
}


def _resolve_colormap(name_or_int):
    """Accept a string name (e.g. 'TURBO') or cv2 int constant."""
    if isinstance(name_or_int, int):
        return name_or_int
    return _COLORMAP_TABLE.get(str(name_or_int).upper(), cv2.COLORMAP_TURBO)


def depth_to_colormap(depth, vmin=0.0, vmax=1.0, colormap=cv2.COLORMAP_TURBO):
    """Return an RGB colormap image from a single-channel depth array."""
    cmap = _resolve_colormap(colormap)
    depth_clipped = np.clip(depth, vmin, vmax)
    if vmax - vmin > 1e-6:
        depth_norm = ((depth_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth_clipped, dtype=np.uint8)
    bgr = cv2.applyColorMap(depth_norm, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _alpha_overlay(base, overlay, alpha=0.55):
    """Blend *overlay* onto *base* with constant alpha (in-place on base)."""
    mask = np.any(overlay != 0, axis=-1)
    base[mask] = (
        base[mask].astype(np.float32) * (1 - alpha)
        + overlay[mask].astype(np.float32) * alpha
    ).astype(np.uint8)


def _draw_pin_marker(canvas, center, color, size=20):
    """Draw a map-pin (location marker) icon at *center* on *canvas*.

    The pin is an inverted teardrop: a filled circle on top with a triangular
    pointer at the bottom, plus a white inner circle to mimic the classic
    map-pin look.
    """
    cx, cy = int(center[0]), int(center[1])
    r = size // 2
    tip_y = cy + int(size * 0.95)

    tri_pts = np.array([
        [cx - int(r * 0.7), cy + int(r * 0.3)],
        [cx + int(r * 0.7), cy + int(r * 0.3)],
        [cx, tip_y],
    ], dtype=np.int32)
    cv2.fillConvexPoly(canvas, tri_pts, color, cv2.LINE_AA)

    cv2.circle(canvas, (cx, cy), r, color, -1, cv2.LINE_AA)

    inner_r = max(2, int(r * 0.42))
    cv2.circle(canvas, (cx, cy), inner_r, (255, 255, 255), -1, cv2.LINE_AA)


def render_depth_comparison(gt_depth, pred_depth, target_h=256, target_w=256):
    """Side-by-side GT vs predicted depth (used by trainer visualization).
    Returns BGR for cv2.imwrite compatibility."""
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

    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "GT Depth", (5, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "WM Pred", (target_w + gap + 5, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    rmse = float(np.sqrt(np.mean((gt_depth - pred_depth) ** 2)))
    cv2.putText(canvas, f"RMSE={rmse:.4f}", (target_w + gap + 5, target_h - 10),
                font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_trajectory_comparison(
    gt_traj, pred_traj, num_humans,
    human_positions=None,
    canvas_size=512, world_range=8.0,
):
    """Standalone GT vs predicted trajectory plot (used by trainer visualization)."""
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
        gt_h, pred_h = gt_traj[h], pred_traj[h]
        if np.all(np.abs(gt_h) > 90):
            continue
        gt_color = _DEFAULT_GT_COLORS[h % len(_DEFAULT_GT_COLORS)]
        pd_color = _DEFAULT_PRED_COLORS[h % len(_DEFAULT_PRED_COLORS)]
        hp = None
        if human_positions is not None and h < len(human_positions):
            _hp = human_positions[h]
            if not np.all(np.abs(_hp) > 90):
                hp = _hp
        if len(pred_h) > 0:
            if hp is not None:
                cv2.line(canvas, world_to_px(hp), world_to_px(pred_h[0]), pd_color, 3, cv2.LINE_AA)
            for t in range(len(pred_h) - 1):
                cv2.line(canvas, world_to_px(pred_h[t]), world_to_px(pred_h[t + 1]), pd_color, 3, cv2.LINE_AA)
            cv2.circle(canvas, world_to_px(pred_h[0]), 6, pd_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, world_to_px(pred_h[-1]), 5, pd_color, 2, cv2.LINE_AA)
        if len(gt_h) > 0:
            if hp is not None:
                cv2.line(gt_layer, world_to_px(hp), world_to_px(gt_h[0]), gt_color, 3, cv2.LINE_AA)
            for t in range(len(gt_h) - 1):
                cv2.line(gt_layer, world_to_px(gt_h[t]), world_to_px(gt_h[t + 1]), gt_color, 3, cv2.LINE_AA)
            for t in range(len(gt_h)):
                pt = world_to_px(gt_h[t])
                r = 6 if t == 0 else (5 if t == len(gt_h) - 1 else 4)
                cv2.circle(gt_layer, pt, r, gt_color, -1, cv2.LINE_AA)

    _alpha_overlay(canvas, gt_layer, alpha=0.6)

    for h in range(min(num_humans, len(gt_traj), len(pred_traj))):
        if np.all(np.abs(gt_traj[h]) > 90):
            continue
        if human_positions is not None and h < len(human_positions):
            hp = human_positions[h]
            if not np.all(np.abs(hp) > 90):
                color = _DEFAULT_GT_COLORS[h % len(_DEFAULT_GT_COLORS)]
                cv2.drawMarker(canvas, world_to_px(hp), color, cv2.MARKER_CROSS, 14, 3, cv2.LINE_AA)

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
            ade_vals.append(np.linalg.norm(gt_traj[h] - pred_traj[h], axis=-1).mean())
        if ade_vals:
            cv2.putText(canvas, f"ADE={float(np.mean(ade_vals)):.3f}m",
                        (canvas_size - 160, canvas_size - 10), font, 0.5, text_color, 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def compose_wm_frame(depth_frame, traj_frame, target_h=256):
    """Horizontally stack depth and trajectory panels (used by trainer visualization)."""
    panels = []
    if depth_frame is not None:
        h_ratio = target_h / depth_frame.shape[0]
        panels.append(cv2.resize(depth_frame, (int(depth_frame.shape[1] * h_ratio), target_h)))
    if traj_frame is not None:
        h_ratio = target_h / traj_frame.shape[0]
        panels.append(cv2.resize(traj_frame, (int(traj_frame.shape[1] * h_ratio), target_h)))
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


def draw_trajectories_on_topdown(
    topdown_map: np.ndarray,
    gt_traj: Optional[np.ndarray],
    pred_traj: Optional[np.ndarray],
    num_humans: int,
    human_positions: Optional[np.ndarray] = None,
    human_goals: Optional[np.ndarray] = None,
    robot_world_pos: Optional[np.ndarray] = None,
    goal_world_pos: Optional[np.ndarray] = None,
    bounds: Optional[Tuple] = None,
    map_shape: Optional[Tuple[int, int]] = None,
    ade: float = 0.0,
    traj_cfg=None,
):
    """Draw GT and predicted future trajectories on a colorized top-down map.

    Trajectories from sensors are in robot-centric 2D coords (delta_x, delta_z).
    We convert them to the same map-pixel coordinate system used by the
    TopDownMap measure so they align with the existing agent paths.

    Args:
        topdown_map: Already-colorized RGB top-down map (may be rotated/resized).
        gt_traj: (N, T, 2) GT future trajectory, robot-centric (x, z).
        pred_traj: (N, T, 2) predicted future trajectory, robot-centric (x, z).
        num_humans: number of valid humans.
        human_positions: (N, 2) current human positions, robot-centric (x, z).
        robot_world_pos: (3,) robot world position [x, y, z] from sim.
        bounds: (lower_bound, upper_bound) from pathfinder.get_bounds().
        map_shape: (H, W) of the *original* (pre-rotation, pre-resize) top-down map.
        ade: average displacement error to display.
    """
    canvas = topdown_map.copy()
    h, w = canvas.shape[:2]

    if gt_traj is None or pred_traj is None:
        return canvas

    has_map_info = (robot_world_pos is not None and bounds is not None
                    and map_shape is not None)

    if has_map_info:
        lower_bound, upper_bound = bounds
        raw_h, raw_w = map_shape  # original map before colorize/rotate/resize
        grid_size_z = abs(upper_bound[2] - lower_bound[2]) / raw_h
        grid_size_x = abs(upper_bound[0] - lower_bound[0]) / raw_w

        rotated = raw_h > raw_w

        if rotated:
            # After rot90: new shape = (raw_w, raw_h) then resized to (h, w)
            scale_x = w / raw_h
            scale_y = h / raw_w
        else:
            # No rotation: shape = (raw_h, raw_w) then resized to (h, w)
            scale_x = w / raw_w
            scale_y = h / raw_h

        def relative_to_px(delta):
            """Convert robot-centric (delta_x, delta_z) to pixel coords on the
            colorized (possibly rotated & resized) map.

            Sensor coords: delta = (world_x - robot_x, world_z - robot_z)
            to_grid maps: world_z -> grid_x (row), world_x -> grid_y (col)
            OpenCV draws at (col, row) = (grid_y, grid_x)
            rot90 CCW: (row, col) -> (orig_cols-1-col, row)
            """
            world_x = robot_world_pos[0] + delta[0]
            world_z = robot_world_pos[2] + delta[1]
            grid_row = (world_z - lower_bound[2]) / grid_size_z
            grid_col = (world_x - lower_bound[0]) / grid_size_x

            if rotated:
                # After rot90: new_row = orig_cols-1-col, new_col = row
                px = int(grid_row * scale_x)
                py = int((raw_w - 1 - grid_col) * scale_y)
            else:
                px = int(grid_col * scale_x)
                py = int(grid_row * scale_y)
            return (np.clip(px, 0, w - 1), np.clip(py, 0, h - 1))
    else:
        cx, cy = w // 2, h // 2
        scale = min(h, w) / 16.0

        def relative_to_px(delta):
            px = int(cx + delta[0] * scale)
            py = int(cy + delta[1] * scale)
            return (np.clip(px, 0, w - 1), np.clip(py, 0, h - 1))

    # ── Read configurable parameters from traj_cfg ──
    draw_gt = _get_traj_cfg_value(traj_cfg, "draw_gt_future", True)
    draw_pred = _get_traj_cfg_value(traj_cfg, "draw_pred_future", True)
    draw_human_goals_flag = _get_traj_cfg_value(traj_cfg, "draw_human_goals", True)
    draw_robot_goal_flag = _get_traj_cfg_value(traj_cfg, "draw_robot_goal", True)
    gt_alpha = float(_get_traj_cfg_value(traj_cfg, "gt_alpha", 0.35))
    pred_alpha = float(_get_traj_cfg_value(traj_cfg, "pred_alpha", 1.0))

    cfg_gt_thick = int(_get_traj_cfg_value(traj_cfg, "gt_thickness", 0))
    cfg_pred_thick = int(_get_traj_cfg_value(traj_cfg, "pred_thickness", 0))
    auto_thickness = max(1, min(h, w) // 150)
    gt_thickness = cfg_gt_thick if cfg_gt_thick > 0 else auto_thickness
    pred_thickness = cfg_pred_thick if cfg_pred_thick > 0 else auto_thickness

    gt_colors = _parse_color_list(
        _get_traj_cfg_value(traj_cfg, "gt_colors", None), _DEFAULT_GT_COLORS)
    pred_colors = _parse_color_list(
        _get_traj_cfg_value(traj_cfg, "pred_colors", None), _DEFAULT_PRED_COLORS)
    robot_goal_color = tuple(
        _get_traj_cfg_value(traj_cfg, "robot_goal_color", _DEFAULT_ROBOT_GOAL_COLOR))

    # ── Draw predicted future trajectories ──
    if pred_alpha < 1.0:
        pred_layer = np.zeros_like(canvas)
    else:
        pred_layer = canvas

    if draw_pred:
        for hi in range(min(num_humans, len(gt_traj), len(pred_traj))):
            pred_h = pred_traj[hi]
            if np.all(np.abs(gt_traj[hi]) > 90):
                continue
            pd_color = pred_colors[hi % len(pred_colors)]

            hp = None
            if human_positions is not None and hi < len(human_positions):
                _hp = human_positions[hi]
                if not np.all(np.abs(_hp) > 90):
                    hp = _hp

            if len(pred_h) > 0:
                if hp is not None:
                    cv2.line(pred_layer, relative_to_px(hp), relative_to_px(pred_h[0]),
                             pd_color, pred_thickness, cv2.LINE_AA)
                for t in range(len(pred_h) - 1):
                    cv2.line(pred_layer, relative_to_px(pred_h[t]),
                             relative_to_px(pred_h[t + 1]),
                             pd_color, pred_thickness, cv2.LINE_AA)

    if draw_pred and pred_alpha < 1.0:
        _alpha_overlay(canvas, pred_layer, alpha=pred_alpha)

    # ── Draw GT future trajectories ──
    gt_layer = np.zeros_like(canvas)

    if draw_gt:
        for hi in range(min(num_humans, len(gt_traj), len(pred_traj))):
            gt_h = gt_traj[hi]
            if np.all(np.abs(gt_h) > 90):
                continue
            gt_color = gt_colors[hi % len(gt_colors)]

            hp = None
            if human_positions is not None and hi < len(human_positions):
                _hp = human_positions[hi]
                if not np.all(np.abs(_hp) > 90):
                    hp = _hp

            if len(gt_h) > 0:
                if hp is not None:
                    cv2.line(gt_layer, relative_to_px(hp), relative_to_px(gt_h[0]),
                             gt_color, gt_thickness, cv2.LINE_AA)
                for t in range(len(gt_h) - 1):
                    cv2.line(gt_layer, relative_to_px(gt_h[t]),
                             relative_to_px(gt_h[t + 1]),
                             gt_color, gt_thickness, cv2.LINE_AA)

    if draw_gt:
        _alpha_overlay(canvas, gt_layer, alpha=gt_alpha)

    # ── Markers ──
    marker_size = max(10, min(h, w) // 30)

    if draw_human_goals_flag:
        for hi in range(min(num_humans, len(gt_traj), len(pred_traj))):
            if np.all(np.abs(gt_traj[hi]) > 90):
                continue
            color = gt_colors[hi % len(gt_colors)]
            if human_goals is not None and hi < len(human_goals):
                goal = human_goals[hi]
                if not np.all(np.abs(goal) > 90):
                    cv2.drawMarker(canvas, relative_to_px(goal), color,
                                   cv2.MARKER_STAR, marker_size, 2, cv2.LINE_AA)

    if draw_robot_goal_flag and goal_world_pos is not None and has_map_info:
        goal_rel = np.array([
            goal_world_pos[0] - robot_world_pos[0],
            goal_world_pos[2] - robot_world_pos[2],
        ])
        _draw_pin_marker(canvas, relative_to_px(goal_rel), robot_goal_color,
                         size=max(14, min(h, w) // 20))

    return canvas


def compose_unified_frame(
    rgb_obs: np.ndarray,
    gt_depth_obs: np.ndarray,
    pred_depth_vis: Optional[np.ndarray],
    topdown_map: Optional[np.ndarray],
    wm_result: Optional[WMStepResult] = None,
    depth_rmse: float = 0.0,
    robot_world_pos: Optional[np.ndarray] = None,
    goal_world_pos: Optional[np.ndarray] = None,
    bounds: Optional[Tuple] = None,
    raw_map_shape: Optional[Tuple[int, int]] = None,
    traj_cfg=None,
):
    """Build the unified eval frame: [metrics bar | RGB | GT Depth | Pred Depth | TopDown+Traj].

    All panels are resized to the same height (the RGB height). A metrics bar
    is drawn above the panels (not overlapping any image). Returns RGB uint8.
    """
    depth_cmap = _resolve_colormap(
        _get_traj_cfg_value(traj_cfg, "depth_colormap", "TURBO"))

    target_h = rgb_obs.shape[0]
    panels: List[np.ndarray] = []

    panels.append(rgb_obs)

    gt_depth_vis = _depth_obs_to_vis(gt_depth_obs, target_h, colormap=depth_cmap)
    panels.append(gt_depth_vis)

    if wm_result is not None and wm_result.pred_depth_raw is not None:
        pred_vis = depth_to_colormap(wm_result.pred_depth_raw, colormap=depth_cmap)
        pred_resized = cv2.resize(pred_vis, (target_h, target_h))
        panels.append(pred_resized)
    elif pred_depth_vis is not None:
        pred_resized = cv2.resize(pred_depth_vis, (target_h, target_h))
        panels.append(pred_resized)

    if topdown_map is not None:
        td = topdown_map.copy()
        if wm_result is not None and wm_result.gt_traj is not None:
            td = draw_trajectories_on_topdown(
                td,
                wm_result.gt_traj, wm_result.pred_traj,
                wm_result.num_humans,
                human_positions=wm_result.human_positions,
                human_goals=wm_result.human_goals,
                robot_world_pos=robot_world_pos,
                goal_world_pos=goal_world_pos,
                bounds=bounds,
                map_shape=raw_map_shape,
                ade=wm_result.traj_ade,
                traj_cfg=traj_cfg,
            )
        h_ratio = target_h / td.shape[0]
        new_w = int(td.shape[1] * h_ratio)
        td = cv2.resize(td, (new_w, target_h))
        panels.append(td)

    gap = 2
    total_w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)

    # ── Build metrics text bar above panels ──
    overlay_cfg = _get_traj_cfg_value(traj_cfg, "overlay_text", None)
    show_text = bool(_get_traj_cfg_value(overlay_cfg, "show", True))

    bar_h = 0
    bar = None
    if show_text:
        bar_h = int(_get_traj_cfg_value(overlay_cfg, "bar_height", 28))
        font_scale = float(_get_traj_cfg_value(overlay_cfg, "font_scale", 0.5))
        text_color = tuple(_get_traj_cfg_value(overlay_cfg, "color", (255, 255, 255)))
        bg_color = tuple(_get_traj_cfg_value(overlay_cfg, "bg_color", (40, 40, 40)))
        text_thick = int(_get_traj_cfg_value(overlay_cfg, "thickness", 1))

        bar = np.full((bar_h, total_w, 3), bg_color, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        parts: List[str] = []
        if wm_result is not None:
            if depth_rmse > 0:
                parts.append(f"Depth RMSE: {depth_rmse:.4f}")
            if wm_result.traj_ade > 0:
                parts.append(f"Traj ADE: {wm_result.traj_ade:.3f}m")
            parts.append(f"Humans: {wm_result.num_humans}")

        if parts:
            text = "  |  ".join(parts)
            text_y = int(bar_h * 0.7)
            cv2.putText(bar, text, (6, text_y), font, font_scale,
                        text_color, text_thick, cv2.LINE_AA)

    # ── Assemble final canvas ──
    canvas = np.zeros((bar_h + target_h, total_w, 3), dtype=np.uint8)
    if bar is not None:
        canvas[:bar_h, :] = bar

    x = 0
    for p in panels:
        canvas[bar_h:, x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    return canvas


def _depth_obs_to_vis(depth_obs: np.ndarray, target_h: int, colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """Convert a raw depth observation (H,W,1) or (H,W) to a colormapped RGB image."""
    d = depth_obs
    if not isinstance(d, np.ndarray):
        d = d.cpu().numpy()
    if d.dtype == np.uint8:
        d = d.astype(np.float32) / 255.0
    if d.ndim == 3:
        d = d[..., 0]
    vis = depth_to_colormap(d, colormap=colormap)
    vis = cv2.resize(vis, (target_h, target_h))
    return vis


class WMVisualizer:
    """
    Stateful helper that runs WM inference each eval step and
    returns structured results for composing the unified eval frame.
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
    def step(self, batch, prev_actions, masks, env_idx) -> Optional[WMStepResult]:
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

        result = WMStepResult()

        # Depth reconstruction
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

            result.pred_depth_raw = pred_depth_np.copy()
            result.pred_depth_vis = depth_to_colormap(pred_depth_np)
            result.depth_rmse = float(np.sqrt(np.mean((gt_depth_np - pred_depth_np) ** 2)))

        # Trajectory prediction
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

            human_positions = None
            human_goals = None
            if human_state_goal_key is not None:
                hsg_np = obs_i[human_state_goal_key][0].cpu().numpy()
                human_positions = hsg_np[:, :2]   # pos_x, pos_z
                human_goals = hsg_np[:, 4:6]      # goal_x, goal_z

            result.gt_traj = gt_traj_raw
            result.pred_traj = pred_traj_np
            result.num_humans = num_humans
            result.human_positions = human_positions
            result.human_goals = human_goals

            valid = min(num_humans, len(gt_traj_raw), len(pred_traj_np))
            ade_vals = []
            for hi in range(valid):
                if np.all(np.abs(gt_traj_raw[hi]) > 90):
                    continue
                disp = np.linalg.norm(gt_traj_raw[hi] - pred_traj_np[hi], axis=-1)
                ade_vals.append(disp.mean())
            if ade_vals:
                result.traj_ade = float(np.mean(ade_vals))

        return result
