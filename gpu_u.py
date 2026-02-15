#!/usr/bin/env python3
"""
简单脚本：维持指定目标的 GPU 利用率。
通过「计算 + 休眠」周期实现；计算阶段用较大矩阵与批量 matmul 拉高实际占用。
按 Ctrl+C 退出。
"""
import time
import argparse

def parse_gpu_ids(gpu, gpus: str, device_count: int):
    if gpus:
        ids = []
        for x in gpus.split(","):
            x = x.strip()
            if not x:
                continue
            ids.append(int(x))
        if not ids:
            raise ValueError("--gpus 不能为空，例如: --gpus 0,1")
        return list(dict.fromkeys(ids))
    if gpu is not None:
        return [gpu]
    return list(range(device_count))

def main():
    parser = argparse.ArgumentParser(description="维持目标 GPU 利用率")
    parser.add_argument("--gpu", type=int, default=None, help="使用的单个 GPU 编号")
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="多 GPU 编号，逗号分隔，例如: 0,1,2（优先于 --gpu；默认自动使用全部 GPU）",
    )
    parser.add_argument("--target", type=float, default=50.0,
                        help="目标利用率百分比 (默认 60)")
    parser.add_argument("--size", type=int, default=2048,
                        help="矩阵边长 (默认 2048，越大 GPU 占用越高)")
    parser.add_argument("--batch", type=int, default=30,
                        help="每轮同步前连续 matmul 次数 (默认 4，提高可减少同步开销)")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("需要 PyTorch。请安装: pip install torch")
        return

    if not torch.cuda.is_available():
        print("未检测到 CUDA GPU")
        return

    device_count = torch.cuda.device_count()
    try:
        gpu_ids = parse_gpu_ids(args.gpu, args.gpus, device_count)
    except ValueError as e:
        print(f"参数错误: {e}")
        return

    invalid_ids = [i for i in gpu_ids if i < 0 or i >= device_count]
    if invalid_ids:
        print(f"GPU 编号无效: {invalid_ids}，当前可用范围: 0 ~ {device_count - 1}")
        return

    devices = [torch.device(f"cuda:{i}") for i in gpu_ids]
    work_ratio = args.target / 100.0
    cycle_sec = 1.0
    work_sec = cycle_sec * work_ratio
    sleep_sec = cycle_sec - work_sec

    n = max(256, args.size)
    batch = max(1, args.batch)
    mats = {}
    for device in devices:
        a = torch.randn(n, n, device=device, dtype=torch.float32)
        b = torch.randn(n, n, device=device, dtype=torch.float32)
        mats[device] = (a, b)

    print(
        f"GPU {gpu_ids}: 目标利用率 ~{args.target}% "
        f"(工作 {work_sec:.2f}s / 休眠 {sleep_sec:.2f}s 每周期), 矩阵 {n}x{n}, batch={batch}"
    )
    print("按 Ctrl+C 退出\n")

    step = 0
    t0 = time.perf_counter()
    try:
        while True:
            step += 1
            end_work = time.perf_counter() + work_sec
            while time.perf_counter() < end_work:
                for _ in range(batch):
                    for device in devices:
                        a, b = mats[device]
                        _ = torch.mm(a, b)
                for device in devices:
                    torch.cuda.synchronize(device)
            time.sleep(sleep_sec)
            if step % 60 == 0:
                elapsed = time.perf_counter() - t0
                print(f"[{elapsed:.0f}s] 运行中, step={step}")
    except KeyboardInterrupt:
        print("\n已退出")

if __name__ == "__main__":
    main()
