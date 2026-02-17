# 实验配置总览

## 配置继承层次

```
exp_01_baseline.yaml (独立) ⭐ BASELINE
  - Falcon baseline
  - 无 World Model
  - 无辅助损失

exp_02_full_wm.yaml (独立) ⭐ MAIN
  - WM encoder: Falcon ResNet50 (统一架构)
  - 完整的 WM 配置
  - 所有其他 WM 实验的基础
  │
  ├── exp_03_no_depth.yaml (消融: No Depth)
  ├── exp_04_no_traj.yaml (消融: No Trajectory)
  │
  ├── exp_09_frozen_wm.yaml (冻结 WM，仅用预训练特征)
  ├── exp_10_dreamer_encoder.yaml (消融: Dreamer CNN encoder)
  │     └── exp_11_frozen_dreamer_wm.yaml (冻结 Dreamer WM)
  │
  └── Pretrained WM 系列 (从 exp_02 继承):
        ├── exp_05_pretrain_ratio_025.yaml (ratio=0.25)
        ├── exp_06_pretrain_ratio_006.yaml (ratio=0.0625)
        ├── exp_07_pretrain_ratio_001.yaml (ratio=0.015625)
        └── exp_08_pretrain_ratio_100.yaml (ratio=1.0)
```

## 实验分组

### 1. Baseline (无 World Model)
- **exp_01_baseline.yaml** ⭐
  - Falcon baseline
  - 无 World Model
  - 无辅助损失
  - ResNet50 从头训练

### 2. 主实验 (Full World Model)
- **exp_02_full_wm.yaml** ⭐
  - WM encoder: Falcon ResNet50 (预训练权重, **冻结**)
  - Policy encoder: ResNet50 (从头训练)
  - Late fusion
  - `freeze_wm_encoder: True` (encoder 冻结，仅训练 RSSM + decoders)
  - ratio = 1.0 (每次 PPO 都训练 WM)
  - epochs_per_update = 15

### 3. 消融实验 (Ablation Studies)
基于 exp_02，移除特定组件：

- **exp_03_no_depth.yaml**
  - 移除 Depth decoder
  - `depth_loss_scale: 0.0`

- **exp_04_no_traj.yaml**
  - 移除 Trajectory prediction
  - `traj_loss_scale: 0.0`

- **exp_10_dreamer_encoder.yaml**
  - WM encoder: Dreamer CNN (从头训练，不冻结)
  - 无预训练权重
  - 对比 Falcon ResNet encoder vs Dreamer CNN encoder

### 4. 冻结 World Model 实验
- **exp_09_frozen_wm.yaml**
  - 预训练 WM: `exp_02/latest.pth` (Falcon encoder)
  - `train_world_model: False` (完全冻结 WM，不做任何 WM 训练)
  - WM 仅作为固定特征提取器，验证预训练 WM 特征的泛化能力

- **exp_11_frozen_dreamer_wm.yaml**
  - 预训练 WM: `exp_10/latest.pth` (Dreamer encoder)
  - `train_world_model: False` (完全冻结 WM)
  - 对比 frozen Falcon WM (exp_09) vs frozen Dreamer WM (exp_11)

### 5. 预训练 World Model 实验
基于 exp_02，使用预训练的 WM，不同训练频率：

- **exp_05_pretrain_ratio_025.yaml**
  - 预训练 WM: `exp_02/latest.pth`
  - ratio = 0.25 (每 4 次 PPO 做 1 次 WM)
  - epochs_per_update = 60

- **exp_06_pretrain_ratio_006.yaml**
  - 预训练 WM: `exp_02/latest.pth`
  - ratio = 0.0625 (每 16 次 PPO 做 1 次 WM)
  - epochs_per_update = 240

- **exp_07_pretrain_ratio_001.yaml**
  - 预训练 WM: `exp_02/latest.pth`
  - ratio = 0.015625 (每 64 次 PPO 做 1 次 WM)
  - epochs_per_update = 966

- **exp_08_pretrain_ratio_100.yaml**
  - 预训练 WM: `exp_02/latest.pth`
  - ratio = 1.0 (每次 PPO 都做 WM)
  - epochs_per_update = 15

## 配置参数对比

| 实验 | WM | Pretrained WM Enc | Freeze Enc | Pretrained WM Ckpt | ratio | epochs | depth | traj |
|------|----|----|----|----|------:|-------:|------:|-----:|
| exp_01 | ❌ | - | - | - | - | - | - | - |
| exp_02 | ✅ | ✅ | 🧊 | - | 1.00 | 15 | ✅ | ✅ |
| exp_03 | ✅ | ✅ | 🧊 | - | 1.00 | 15 | ❌ | ✅ |
| exp_04 | ✅ | ✅ | 🧊 | - | 1.00 | 15 | ✅ | ❌ |
| exp_10 | ✅ | ❌ Dreamer | ❌ | - | 1.00 | 15 | ✅ | ✅ |
| exp_11 | ✅ | ❌ Dreamer | 🧊 | ✅ (exp_10) | 🧊 all frozen | - | ✅ | ✅ |
| exp_09 | ✅ | ✅ | 🧊 | ✅ (exp_02) | 🧊 all frozen | - | ✅ | ✅ |
| exp_05 | ✅ | ✅ | 🧊 | ✅ | 0.25 | 60 | ✅ | ✅ |
| exp_06 | ✅ | ✅ | 🧊 | ✅ | 0.0625 | 240 | ✅ | ✅ |
| exp_07 | ✅ | ✅ | 🧊 | ✅ | 0.015625 | 966 | ✅ | ✅ |
| exp_08 | ✅ | ✅ | 🧊 | ✅ | 1.00 | 15 | ✅ | ✅ |

## 运行顺序

1. **首先运行**: `exp_01` (baseline) 和 `exp_02` (main)
2. **然后运行**: `exp_03`, `exp_04`, `exp_10` (消融实验)
3. **最后运行**: `exp_05`~`exp_09` (需要 exp_02 的 checkpoint), `exp_11` (需要 exp_10 的 checkpoint)

## 预期研究问题

1. **WM 是否有效？** → 对比 exp_01 vs exp_02
2. **Depth 重要性？** → 对比 exp_02 vs exp_03
3. **Trajectory 重要性？** → 对比 exp_02 vs exp_04
4. **预训练 WM 是否有帮助？** → 对比 exp_02 vs exp_08
5. **WM 训练频率影响？** → 对比 exp_05, exp_06, exp_07, exp_08
6. **冻结 WM 是否足够？** → 对比 exp_02 vs exp_09 (训练 vs 冻结)
7. **Encoder 架构影响？** → 对比 exp_02 vs exp_10 (Falcon ResNet vs Dreamer CNN)
8. **Frozen WM: Falcon vs Dreamer？** → 对比 exp_09 vs exp_11
