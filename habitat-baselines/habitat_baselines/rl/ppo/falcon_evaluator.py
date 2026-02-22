import os
from collections import defaultdict
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import tqdm

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat.utils.visualizations import maps
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from habitat_baselines.utils.wm_visualizer import (
    WMVisualizer,
    WMStepResult,
    compose_unified_frame,
    _depth_obs_to_vis,
)

import json

class FALCONEvaluator(Evaluator):
    """
    Only difference is record the success rate of each episode while evaluating.
    Similar to ORCAEvaluator.
    """

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        success_cal = 0 ## my added
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        # ── World Model visualisation ──
        wm_obj = None
        for candidate in [
            getattr(agent, "actor_critic", None),
            getattr(getattr(agent, "actor_critic", None), "_active_policies", [None])[0]
            if hasattr(getattr(agent, "actor_critic", None), "_active_policies") else None,
            getattr(agent._agents[0], "_actor_critic", None) if hasattr(agent, "_agents") else None,
        ]:
            _net = getattr(candidate, "net", None) if candidate is not None else None
            if _net is not None and hasattr(_net, "world_model") and _net.world_model is not None:
                wm_obj = _net.world_model
                break
        save_images = getattr(config.habitat_baselines.eval, "save_images", False)
        collect_frames = len(config.habitat_baselines.eval.video_option) > 0 or save_images
        has_wm = wm_obj is not None and collect_frames
        if has_wm:
            wm_vis = WMVisualizer(wm_obj, device, config.habitat_baselines.num_environments)
        else:
            wm_vis = None

        if collect_frames:
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}
                    )
                ]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
        else:
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)
        if save_images:
            image_dir = getattr(config.habitat_baselines.eval, "image_dir", None) or os.path.join(config.habitat_baselines.video_dir, "images")
            os.makedirs(image_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        actions_record = defaultdict(list)
        agent.eval()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if hasattr(agent, '_agents') and agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                step_data = [a.numpy() for a in action_data.env_actions.cpu()]
            elif is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            for i in range(envs.num_envs):
                episode_key = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                    ep_eval_count[
                        (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ]
                )

                action_value = step_data[i]
                if isinstance(action_value, np.ndarray):
                    stored_action = {
                        "type": "array",
                        "value": action_value.tolist()
                    }
                else:
                    stored_action = {
                        "type": "array",
                        "value": np.array(action_value).tolist()
                    }

                actions_record[episode_key].append(stored_action)

            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            # ── Collect WM results per env ──
            wm_results = {}
            if wm_vis is not None:
                for i in range(envs.num_envs):
                    wm_results[i] = wm_vis.step(batch, prev_actions, not_done_masks, i)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if collect_frames:
                    wm_res_i = wm_results.get(i, None)

                    if wm_vis is not None:
                        # ── Unified layout: [RGB | GT Depth | Pred Depth | TopDown+Traj] ──
                        obs_i = {k: v[i] for k, v in batch.items()}
                        rgb_key = next(
                            (k for k in obs_i if "rgb" in k.lower() and len(obs_i[k].shape) > 1), None
                        )
                        depth_key = next(
                            (k for k in obs_i if "depth" in k.lower() and len(obs_i[k].shape) > 1), None
                        )

                        rgb_obs = obs_i[rgb_key] if rgb_key else None
                        gt_depth_obs = obs_i[depth_key] if depth_key else None

                        if rgb_obs is not None:
                            if not isinstance(rgb_obs, np.ndarray):
                                rgb_obs = rgb_obs.cpu().numpy()
                            if rgb_obs.dtype != np.uint8:
                                rgb_obs = (rgb_obs * 255.0).astype(np.uint8)

                        if gt_depth_obs is not None:
                            if not isinstance(gt_depth_obs, np.ndarray):
                                gt_depth_obs = gt_depth_obs.cpu().numpy()

                        topdown_map = None
                        robot_world_pos = None
                        goal_world_pos = None
                        td_bounds = None
                        raw_map_shape = None
                        traj_vis_cfg = None
                        if "top_down_map" in disp_info:
                            td_info = disp_info["top_down_map"]
                            raw_map_shape = td_info["map"].shape[:2]
                            topdown_map = maps.colorize_draw_agent_and_fit_to_height(
                                td_info, rgb_obs.shape[0] if rgb_obs is not None else 256
                            )
                            td_bounds = td_info.get("bounds", None)
                            rwp = td_info.get("robot_world_pos", None)
                            if rwp is not None:
                                robot_world_pos = np.asarray(rwp, dtype=np.float64)
                            gwp = td_info.get("goal_world_pos", None)
                            if gwp is not None:
                                goal_world_pos = np.asarray(gwp, dtype=np.float64)
                            traj_vis_cfg = td_info.get("trajectory_vis", None)

                        pred_depth_vis = wm_res_i.pred_depth_vis if wm_res_i else None
                        depth_rmse = wm_res_i.depth_rmse if wm_res_i else 0.0

                        compose_kwargs = dict(
                            wm_result=wm_res_i, depth_rmse=depth_rmse,
                            robot_world_pos=robot_world_pos,
                            goal_world_pos=goal_world_pos,
                            bounds=td_bounds, raw_map_shape=raw_map_shape,
                            traj_cfg=traj_vis_cfg,
                        )

                        if not not_done_masks[i].any().item():
                            black_rgb = np.zeros_like(rgb_obs) if rgb_obs is not None else None
                            black_depth = np.zeros_like(gt_depth_obs) if gt_depth_obs is not None else None
                            final_frame = compose_unified_frame(
                                black_rgb, black_depth, pred_depth_vis, topdown_map,
                                **compose_kwargs,
                            )
                            final_frame = overlay_frame(final_frame, disp_info)
                            rgb_frames[i].append(final_frame)

                            frame = compose_unified_frame(
                                rgb_obs, gt_depth_obs, pred_depth_vis, topdown_map,
                                **compose_kwargs,
                            )
                            rgb_frames[i].append(frame)
                        else:
                            frame = compose_unified_frame(
                                rgb_obs, gt_depth_obs, pred_depth_vis, topdown_map,
                                **compose_kwargs,
                            )
                            frame = overlay_frame(frame, disp_info)
                            rgb_frames[i].append(frame)
                    else:
                        # ── Fallback: original layout (no WM) ──
                        frame = observations_to_image(
                            {k: v[i] for k, v in batch.items()}, disp_info
                        )
                        if not not_done_masks[i].any().item():
                            final_frame = observations_to_image(
                                {k: v[i] * 0.0 for k, v in batch.items()},
                                disp_info,
                            )
                            final_frame = overlay_frame(final_frame, disp_info)
                            rgb_frames[i].append(final_frame)
                            rgb_frames[i].append(frame)
                        else:
                            frame = overlay_frame(frame, disp_info)
                            rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].any().item():
                    pbar.update()
                    if "success" in disp_info:
                        success_cal += disp_info['success']
                        print(f"Till now Success Rate: {success_cal/(len(stats_episodes)+1)}")
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        print(f"This is Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}.")
                        
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            images=rgb_frames[i][:-1],
                            scene_id=f"{current_episodes_info[i].scene_id}".split('/')[-1].split('.')[0],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        if wm_vis is not None:
                            wm_vis.reset_env(i)

                    if save_images and rgb_frames is not None:
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        ep_id = f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}"
                        image_dir = getattr(config.habitat_baselines.eval, "image_dir", None)
                        if image_dir is None or image_dir == "":
                            image_dir = os.path.join(config.habitat_baselines.video_dir, "images")
                        out_dir = os.path.join(image_dir, scene_id, ep_id)
                        os.makedirs(out_dir, exist_ok=True)
                        frames_to_save = rgb_frames[i][:-1]
                        for t, frame in enumerate(frames_to_save):
                            path = os.path.join(out_dir, f"frame_{t:06d}.png")
                            if frame.ndim == 3 and frame.shape[2] == 3:
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(path, frame)

                    if collect_frames:
                        rgb_frames[i] = rgb_frames[i][-1:]

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        # ==== 保存 result.json ====
        result_path = os.path.join("output/", "result.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        evalai_result = {
                            "SR": round(aggregated_stats.get("success", 0), 4),
                            "SPL": round(aggregated_stats.get("spl", 0), 4),
                            "PSC": round(aggregated_stats.get("psc", 0), 4),
                            "H-Coll": round(aggregated_stats.get("human_collision", 0), 4),
                            "Total": round(
                                0.4 * aggregated_stats.get("success", 0)
                                + 0.3 * aggregated_stats.get("spl", 0)
                                + 0.3 * aggregated_stats.get("psc", 0),
                                4,
                                    ),
                        }

        with open(result_path, "w") as f:
            json.dump(evalai_result, f, indent=2)

        # ==== 保存 actions.json ====
        actions_output_path = os.path.join("output/", "actions.json")
        os.makedirs(os.path.dirname(actions_output_path), exist_ok=True)
        serializable_actions = {
            f"{scene_id}|{episode_id}|{eval_count}": actions
            for (scene_id, episode_id, eval_count), actions in actions_record.items()
        }
        with open(actions_output_path, "w") as f:
            json.dump(serializable_actions, f, indent=2)