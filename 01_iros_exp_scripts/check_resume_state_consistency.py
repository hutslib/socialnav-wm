#!/usr/bin/env python3
"""
检查 .habitat-resume-state.pth 是否可用于“指标连贯续训”。

用法:
  # 仅检查 resume_state 自身结构与关键字段
  python 01_iros_exp_scripts/check_resume_state_consistency.py \
    --resume-state experiments/exp_09_baseline_no_aux/checkpoints/.habitat-resume-state.pth

  # 额外与指定 ckpt 做一致性校验（推荐）
  python 01_iros_exp_scripts/check_resume_state_consistency.py \
    --resume-state experiments/exp_09_baseline_no_aux/checkpoints/.habitat-resume-state.pth \
    --ckpt experiments/exp_09_baseline_no_aux/checkpoints/ckpt.128.pth

说明:
- 若 resume_state 的 requeue_stats/window_episode_stats 缺失或为空，续训后
  success/spl 等窗口指标常会出现明显跳变。
- 若与 ckpt 的 step/update 对不上，通常会出现“步数看起来接上，但指标不连贯”。
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Any, Dict, Optional

import torch


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _err(msg: str) -> None:
    print(f"[ERROR] {msg}")


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _safe_get_total_steps(cfg: Any) -> Optional[float]:
    try:
        return float(cfg.habitat_baselines.total_num_steps)
    except Exception:
        return None


def _infer_update_from_step(cfg: Any, step: Optional[int]) -> Optional[int]:
    if step is None:
        return None
    try:
        num_envs = int(cfg.habitat_baselines.num_environments)
        ppo_steps = int(cfg.habitat_baselines.rl.ppo.num_steps)
        denom = num_envs * ppo_steps
        if denom <= 0:
            return None
        return step // denom
    except Exception:
        return None


def _safe_get_local_steps_per_update(cfg: Any) -> Optional[int]:
    """单进程每次 update 的理论步数（num_envs * ppo_num_steps）。"""
    try:
        num_envs = int(cfg.habitat_baselines.num_environments)
        ppo_steps = int(cfg.habitat_baselines.rl.ppo.num_steps)
        denom = num_envs * ppo_steps
        if denom <= 0:
            return None
        return denom
    except Exception:
        return None


def _len_of_window_count(window_stats: Dict[str, Any]) -> Optional[int]:
    count = window_stats.get("count")
    if count is None:
        return None
    try:
        return len(count)
    except Exception:
        return None


_PLACEHOLDER_CACHE: Dict[tuple[str, str], type] = {}


def _placeholder_type(module: str, name: str) -> type:
    key = (module, name)
    if key not in _PLACEHOLDER_CACHE:
        cls_name = f"Placeholder_{module.replace('.', '_')}_{name}"
        _PLACEHOLDER_CACHE[key] = type(cls_name, (), {})
    return _PLACEHOLDER_CACHE[key]


class _LenientUnpickler(pickle.Unpickler):
    """在缺失/重型模块时用占位类型，避免 torch.load 直接失败。"""

    _heavy_prefixes = {
        "falcon",
        "habitat",
        "habitat_baselines",
        "habitat_sim",
        "gym",
        "magnum",
    }

    def find_class(self, module: str, name: str) -> Any:
        top = module.split(".", 1)[0]
        if top in self._heavy_prefixes:
            return _placeholder_type(module, name)
        try:
            return super().find_class(module, name)
        except Exception:
            return _placeholder_type(module, name)


class _LenientPickleModule:
    """提供 torch.load 所需的 pickle_module 接口。"""

    __name__ = "lenient_pickle_module"
    Unpickler = _LenientUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    UnpicklingError = pickle.UnpicklingError


_LENIENT_PICKLE = _LenientPickleModule


def _load_pth(path: str) -> Dict[str, Any]:
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_LENIENT_PICKLE,
    )
    if not isinstance(obj, dict):
        raise TypeError(f"文件不是 dict: {path}")
    return obj


def check_resume_state(
    resume_state: Dict[str, Any],
    strict_window: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []

    # 注意：resume payload 由 self._agent.get_resume_state() 展开而来，不一定有 "state_dict" 键名。
    for k in ("config", "requeue_stats"):
        if k not in resume_state:
            errors.append(f"resume_state 缺少关键字段: {k}")

    rq = resume_state.get("requeue_stats")
    if not isinstance(rq, dict):
        errors.append("resume_state['requeue_stats'] 不是 dict")
        return errors, warns

    for k in (
        "num_steps_done",
        "num_updates_done",
        "count_checkpoints",
        "running_episode_stats",
        "window_episode_stats",
    ):
        if k not in rq:
            errors.append(f"requeue_stats 缺少字段: {k}")

    num_steps_done = _to_int(rq.get("num_steps_done"))
    num_updates_done = _to_int(rq.get("num_updates_done"))
    if num_steps_done is None or num_steps_done < 0:
        errors.append(f"num_steps_done 非法: {rq.get('num_steps_done')}")
    if num_updates_done is None or num_updates_done < 0:
        errors.append(f"num_updates_done 非法: {rq.get('num_updates_done')}")

    running_stats = rq.get("running_episode_stats")
    if not isinstance(running_stats, dict):
        errors.append("running_episode_stats 不是 dict")
    else:
        for k in ("count", "reward"):
            if k not in running_stats:
                warns.append(f"running_episode_stats 缺少 {k}")

    window_stats = rq.get("window_episode_stats")
    if not isinstance(window_stats, dict):
        errors.append("window_episode_stats 不是 dict")
    else:
        if len(window_stats) == 0:
            msg = "window_episode_stats 为空（续训后窗口指标大概率跳变）"
            if strict_window:
                errors.append(msg)
            else:
                warns.append(msg)
        cnt_len = _len_of_window_count(window_stats)
        if cnt_len == 0:
            msg = "window_episode_stats['count'] 长度为 0（窗口统计已清空）"
            if strict_window:
                errors.append(msg)
            else:
                warns.append(msg)
        elif cnt_len is None:
            warns.append("window_episode_stats['count'] 不存在或不可取长度")

    cfg = resume_state.get("config")
    if cfg is not None and num_steps_done is not None:
        total_steps = _safe_get_total_steps(cfg)
        if total_steps is not None:
            if num_steps_done > total_steps * 1.01:
                warns.append(
                    f"num_steps_done({num_steps_done}) 大于 total_num_steps({total_steps})，请确认配置是否一致"
                )
            else:
                _ok(
                    f"step 进度: {num_steps_done}/{int(total_steps)} ({100.0 * num_steps_done / total_steps:.2f}%)"
                )

        infer_update = _infer_update_from_step(cfg, num_steps_done)
        if infer_update is not None and num_updates_done is not None:
            local_denom = _safe_get_local_steps_per_update(cfg)
            # 在分布式训练中，global step / update 往往约等于 local_denom * world_size，
            # 若直接用 local_denom 推断 update 会严重偏大，属于正常现象，不应当告警。
            if local_denom is not None and num_updates_done > 0:
                implied_step_per_update = num_steps_done / float(num_updates_done)
                scale = implied_step_per_update / float(local_denom)
                if scale > 1.5:
                    _ok(
                        f"检测到分布式步数缩放（global/local 约 {scale:.2f}x），跳过本地口径 update 一致性告警"
                    )
                else:
                    # 单卡或近似单卡口径下，允许少量 update 偏差
                    if abs(infer_update - num_updates_done) > 2:
                        warns.append(
                            f"num_updates_done({num_updates_done}) 与按 step 推断值({infer_update}) 差异较大"
                        )
            else:
                if abs(infer_update - num_updates_done) > 2:
                    warns.append(
                        f"num_updates_done({num_updates_done}) 与按 step 推断值({infer_update}) 差异较大"
                    )

    return errors, warns


def check_against_ckpt(
    resume_state: Dict[str, Any],
    ckpt: Dict[str, Any],
    strict_window: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []

    rq = resume_state.get("requeue_stats", {})
    if not isinstance(rq, dict):
        errors.append("resume_state.requeue_stats 非 dict，无法与 ckpt 对比")
        return errors, warns

    rs_step = _to_int(rq.get("num_steps_done"))
    rs_update = _to_int(rq.get("num_updates_done"))

    ckpt_rq = ckpt.get("requeue_stats")
    if isinstance(ckpt_rq, dict):
        ck_step = _to_int(ckpt_rq.get("num_steps_done"))
        ck_update = _to_int(ckpt_rq.get("num_updates_done"))
        if rs_step is not None and ck_step is not None and rs_step != ck_step:
            errors.append(
                f"num_steps_done 不一致: resume={rs_step}, ckpt={ck_step}"
            )
        if rs_update is not None and ck_update is not None and rs_update != ck_update:
            errors.append(
                f"num_updates_done 不一致: resume={rs_update}, ckpt={ck_update}"
            )

        rs_ws = rq.get("window_episode_stats", {})
        ck_ws = ckpt_rq.get("window_episode_stats", {})
        if isinstance(rs_ws, dict) and isinstance(ck_ws, dict):
            rs_keys = set(rs_ws.keys())
            ck_keys = set(ck_ws.keys())
            if rs_keys != ck_keys:
                warns.append(
                    f"window_episode_stats 键集合不一致: resume={len(rs_keys)} keys, ckpt={len(ck_keys)} keys"
                )
            rs_cnt = _len_of_window_count(rs_ws)
            ck_cnt = _len_of_window_count(ck_ws)
            if rs_cnt is not None and ck_cnt is not None and rs_cnt != ck_cnt:
                msg = (
                    f"window count 长度不一致: resume={rs_cnt}, ckpt={ck_cnt}"
                )
                if strict_window:
                    errors.append(msg)
                else:
                    warns.append(msg)
        _ok("已使用 ckpt.requeue_stats 完整对比 step/update/window")
        return errors, warns

    # 旧格式 ckpt: 没有 requeue_stats，只能用 extra_state.step 做弱校验
    extra = ckpt.get("extra_state", {})
    if isinstance(extra, dict):
        ck_step = _to_int(extra.get("step"))
        if ck_step is not None and rs_step is not None and ck_step != rs_step:
            warns.append(
                f"旧格式 ckpt 仅 step 可比，发现不一致: resume={rs_step}, ckpt.extra_state.step={ck_step}"
            )
        else:
            _ok("旧格式 ckpt 的 extra_state.step 与 resume step 一致")
    else:
        warns.append("ckpt 无 requeue_stats 且无 extra_state，无法做有效对比")

    return errors, warns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-state",
        required=True,
        help="Path to .habitat-resume-state.pth",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="可选：要对比的 ckpt 路径（推荐）",
    )
    parser.add_argument(
        "--strict-window",
        action="store_true",
        help="将 window 为空/长度异常视为错误（默认仅警告）",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="若存在警告也返回非 0",
    )
    args = parser.parse_args()

    resume_path = os.path.abspath(args.resume_state)
    if not os.path.isfile(resume_path):
        _err(f"resume_state 文件不存在: {resume_path}")
        return 2

    ckpt_path = None
    if args.ckpt is not None:
        ckpt_path = os.path.abspath(args.ckpt)
        if not os.path.isfile(ckpt_path):
            _err(f"ckpt 文件不存在: {ckpt_path}")
            return 2

    print("==== Resume State Consistency Check ====")
    print(f"resume_state: {resume_path}")
    if ckpt_path:
        print(f"ckpt:         {ckpt_path}")

    try:
        resume_state = _load_pth(resume_path)
    except Exception as e:
        _err(f"加载 resume_state 失败: {e}")
        return 2

    errors, warns = check_resume_state(
        resume_state=resume_state,
        strict_window=args.strict_window,
    )

    if ckpt_path is not None:
        try:
            ckpt = _load_pth(ckpt_path)
        except Exception as e:
            _err(f"加载 ckpt 失败: {e}")
            return 2
        e2, w2 = check_against_ckpt(
            resume_state=resume_state,
            ckpt=ckpt,
            strict_window=args.strict_window,
        )
        errors.extend(e2)
        warns.extend(w2)

    print("")
    if errors:
        for m in errors:
            _err(m)
    if warns:
        for m in warns:
            _warn(m)

    if not errors and not warns:
        _ok("检查通过：未发现结构或一致性问题")
    elif not errors:
        _ok("检查完成：无致命错误，但有警告")

    if errors:
        print("RESULT: FAIL")
        return 2
    if warns and args.fail_on_warning:
        print("RESULT: WARN_AS_FAIL")
        return 3
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
