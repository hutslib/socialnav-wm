#!/usr/bin/env python3
"""
从 TensorBoard 日志绘制多实验对比曲线：reward / SPL / Success vs step。

用法:
  # 绘制 experiments/ 下所有 exp_* 的 tb 目录
  python 01_iros_exp_scripts/plot_tensorboard_curves.py

  # 指定实验目录或名称
  python 01_iros_exp_scripts/plot_tensorboard_curves.py --experiments exp_01_baseline_falcon exp_02_full_wm

  # 指定 experiments 根目录
  python 01_iros_exp_scripts/plot_tensorboard_curves.py --root /path/to/experiments
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _get_event_dirs(log_dir: Path):
    """返回该目录下所有包含 event 文件的（子）目录。"""
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    if list(log_dir.glob("events.out.tfevents.*")):
        return [log_dir]
    out = []
    for sub in sorted(log_dir.iterdir()):
        if sub.is_dir() and list(sub.glob("events.out.tfevents.*")):
            out.append(sub)
    return out


def _collect_event_files(tb_dirs: list[Path]) -> list[Path]:
    """收集多个 tb 目录下的所有 event 文件路径（用于逐文件读取，避免 purge）。"""
    files = []
    for d in tb_dirs:
        d = Path(d)
        files.extend(d.glob("events.out.tfevents.*"))
        for sub in d.iterdir():
            if sub.is_dir():
                files.extend(sub.glob("events.out.tfevents.*"))
    return sorted(files, key=lambda p: (p.stat().st_mtime, str(p)))


def _load_scalars_from_event_files(
    event_files: list[Path], tags: list[str]
) -> dict[str, tuple[list, list]]:
    """逐 event 文件读取 scalar，避免目录级读取触发 out-of-order purge。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {tag: ([], []) for tag in tags}
    # 我们请求的 tag -> 文件中可能出现的 tag 名（部分实验写 metrics/spl，部分写 eval_metrics/spl）
    tag_to_file_tags = {}
    for t in tags:
        tag_to_file_tags[t] = [t]
        if t.startswith("eval_metrics/"):
            tag_to_file_tags[t].append("metrics/" + t.replace("eval_metrics/", "", 1))
        elif t.startswith("metrics/"):
            tag_to_file_tags[t].append("eval_metrics/" + t.replace("metrics/", "", 1))
    all_steps = {tag: [] for tag in tags}
    all_values = {tag: [] for tag in tags}
    for ef in event_files:
        try:
            ea = EventAccumulator(str(ef))
            ea.Reload()
            scalar_tags = set(ea.Tags().get("scalars", []))
            for our_tag in tags:
                for file_tag in tag_to_file_tags[our_tag]:
                    if file_tag not in scalar_tags:
                        continue
                    for e in ea.Scalars(file_tag):
                        all_steps[our_tag].append(e.step)
                        all_values[our_tag].append(e.value)
        except Exception:
            continue
    result = {}
    for tag in tags:
        steps, values = all_steps[tag], all_values[tag]
        if not steps:
            result[tag] = ([], [])
            continue
        steps = np.array(steps)
        values = np.array(values)
        order = np.argsort(steps, kind="stable")
        steps = steps[order]
        values = values[order]
        _, rev_first = np.unique(steps[::-1], return_index=True)
        last_idx = len(steps) - 1 - rev_first
        result[tag] = (steps[last_idx].tolist(), values[last_idx].tolist())
    return result


def load_tb_scalars(log_dir: str | Path, tags: list[str]):
    """从单个 TensorBoard 目录读取指定 tag 的 (step, value)，支持多 event 文件。"""
    log_dir = Path(log_dir)
    event_dirs = _get_event_dirs(log_dir)
    if not event_dirs:
        return {tag: ([], []) for tag in tags}
    event_files = _collect_event_files(event_dirs)
    if not event_files:
        return {tag: ([], []) for tag in tags}
    return _load_scalars_from_event_files(event_files, tags)


def find_all_tb_dirs_under(exp_dir: Path) -> list[Path]:
    """找出实验目录下所有 TensorBoard 目录（tb、tb_xxx 等，用于合并断点续训的多次 run）。"""
    exp_dir = Path(exp_dir)
    if not exp_dir.is_dir():
        return []
    found = []
    for d in exp_dir.iterdir():
        if not d.is_dir():
            continue
        # 只认名为 tb 或 tb_* 或含 tensorboard 的子目录，且其中要有 event 文件
        if (d.name == "tb" or d.name.startswith("tb_") or "tensorboard" in d.name.lower()) and _get_event_dirs(d):
            found.append(d)
    found.sort(key=lambda p: p.name)
    return found


def load_tb_scalars_merged(tb_dirs: list[Path], tags: list[str]) -> dict:
    """从多个 TensorBoard 目录读取并合并 scalar（逐 event 文件读，避免 purge）。"""
    if not tb_dirs:
        return {tag: ([], []) for tag in tags}
    event_files = _collect_event_files(tb_dirs)
    if not event_files:
        return {tag: ([], []) for tag in tags}
    return _load_scalars_from_event_files(event_files, tags)


def smooth(y: list[float], weight: float = 0.9) -> np.ndarray:
    """指数滑动平均。"""
    if not y:
        return np.array(y)
    out = [y[0]]
    for v in y[1:]:
        out.append(weight * out[-1] + (1 - weight) * v)
    return np.array(out)


def plot_curves(
    data_by_exp: dict[str, dict],
    tag: str,
    ylabel: str,
    title: str,
    out_path: str | Path,
    smooth_weight: float = 0.0,
):
    """画单张图：多实验同一 tag 的曲线。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for exp_name, scalars in data_by_exp.items():
        if tag not in scalars:
            continue
        steps, values = scalars[tag]
        if not steps:
            continue
        steps = np.array(steps)
        values = np.array(values)
        if smooth_weight > 0:
            values = smooth(values.tolist(), weight=smooth_weight)
        ax.plot(steps, values, label=exp_name, alpha=0.85)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def plot_spl_success_compare(
    data_by_exp: dict[str, dict],
    out_path: str | Path,
):
    """一张图内两个子图：SPL vs step 与 Success vs step 对比。"""
    fig, (ax_spl, ax_sr) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for exp_name, scalars in data_by_exp.items():
        for tag, ax, ylabel in [
            ("eval_metrics/spl", ax_spl, "SPL"),
            ("eval_metrics/success", ax_sr, "Success Rate"),
        ]:
            if tag not in scalars:
                continue
            steps, values = scalars[tag]
            if not steps:
                continue
            steps = np.array(steps)
            values = np.array(values)
            ax.plot(steps, values, label=exp_name, alpha=0.85)
    ax_spl.set_ylabel("SPL")
    ax_spl.set_title("Eval SPL vs Step")
    if ax_spl.get_legend_handles_labels()[0]:
        ax_spl.legend(loc="best", fontsize=8)
    ax_spl.grid(True, alpha=0.3)
    ax_sr.set_xlabel("Step")
    ax_sr.set_ylabel("Success Rate")
    ax_sr.set_title("Eval Success Rate vs Step")
    if ax_sr.get_legend_handles_labels()[0]:
        ax_sr.legend(loc="best", fontsize=8)
    ax_sr.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="从 TensorBoard 绘制 reward / SPL vs step 对比曲线"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="experiments",
        help="实验根目录，下有 exp_xxx/tb",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="指定实验名（如 exp_01_baseline_falcon），不指定则自动扫描 root 下所有 exp_*/tb",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/plots",
        help="图片输出目录",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="reward 曲线平滑系数 (0~1)，0 表示不平滑。建议 0.6~0.9",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 每个实验可能对应多个 tb 目录（断点续训），收集 (exp_name, list of tb dirs)
    if args.experiments:
        exp_names = list(args.experiments)
        exp_to_tb_dirs = []
        for name in exp_names:
            exp_dir = root / name
            dirs = find_all_tb_dirs_under(exp_dir)
            if not dirs:
                dirs = [exp_dir / "tb"] if (exp_dir / "tb").exists() else []
            exp_to_tb_dirs.append((name, dirs))
    else:
        exp_to_tb_dirs = []
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name.startswith("exp_"):
                dirs = find_all_tb_dirs_under(d)
                if dirs:
                    exp_to_tb_dirs.append((d.name, dirs))

    if not exp_to_tb_dirs:
        print("未找到任何实验的 TensorBoard 目录。")
        return

    for name, dirs in exp_to_tb_dirs:
        if len(dirs) > 1:
            try:
                rel = [str(p.relative_to(root)) for p in dirs]
            except ValueError:
                rel = [p.name for p in dirs]
            print(f"  合并 {name}: {len(dirs)} 个 tb 目录 -> {rel}")

    # 训练 reward: 标量名为 "reward"
    # 评估: eval_metrics/spl, eval_metrics/success, eval_reward/average_reward
    tags_train = ["reward"]
    tags_eval = [
        "eval_metrics/spl",
        "eval_metrics/success",
        "eval_reward/average_reward",
    ]
    all_tags = list(set(tags_train + tags_eval))

    data_by_exp = {}
    for name, tb_dir_list in exp_to_tb_dirs:
        data_by_exp[name] = load_tb_scalars_merged(tb_dir_list, all_tags)

    # Reward vs step（训练曲线，可平滑）
    plot_curves(
        data_by_exp,
        tag="reward",
        ylabel="Reward",
        title="Training Reward vs Step",
        out_path=out_dir / "reward_vs_step.png",
        smooth_weight=args.smooth,
    )

    # SPL vs step（评估曲线）
    plot_curves(
        data_by_exp,
        tag="eval_metrics/spl",
        ylabel="SPL",
        title="Eval SPL vs Step",
        out_path=out_dir / "spl_vs_step.png",
        smooth_weight=0.0,
    )

    # Success Rate vs step（评估曲线）
    plot_curves(
        data_by_exp,
        tag="eval_metrics/success",
        ylabel="Success Rate",
        title="Eval Success Rate vs Step",
        out_path=out_dir / "success_vs_step.png",
        smooth_weight=0.0,
    )

    # SPL 与 Success 对比（一张图两子图）
    plot_spl_success_compare(
        data_by_exp,
        out_path=out_dir / "spl_success_vs_step.png",
    )

    # 可选：评估阶段 average reward
    plot_curves(
        data_by_exp,
        tag="eval_reward/average_reward",
        ylabel="Eval Average Reward",
        title="Eval Average Reward vs Step",
        out_path=out_dir / "eval_reward_vs_step.png",
        smooth_weight=0.0,
    )


if __name__ == "__main__":
    main()
