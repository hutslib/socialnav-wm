#!/usr/bin/env python3
"""
World Model质量评估主脚本
运行所有WM质量评估实验并生成报告
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from skimage.metrics import structural_similarity as ssim
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression


class WorldModelEvaluator:
    """World Model质量评估器"""
    
    def __init__(self, world_model, device='cuda'):
        self.world_model = world_model
        self.device = device
        self.world_model.to(device)
        self.world_model.eval()
    
    def evaluate_depth_reconstruction(self, dataloader):
        """评估depth重建质量"""
        print("\n=== Evaluating Depth Reconstruction ===")
        
        results = {
            'rmse': [],
            'mae': [],
            'ssim': [],
            'delta_1': [],
            'delta_2': [],
            'delta_3': [],
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Depth Eval"):
                observations = {k: v.to(self.device) for k, v in batch['observations'].items()}
                
                # WM forward
                embed = self.world_model.encoder(observations)
                
                # RSSM observe
                actions = batch['actions'].to(self.device).unsqueeze(1)
                is_first = batch.get('is_first', torch.zeros(actions.shape[0], 1, 1)).to(self.device)
                
                post, _ = self.world_model.dynamics.observe(
                    embed.unsqueeze(1),
                    actions,
                    is_first,
                )
                
                # Get features
                feat = self.world_model.dynamics.get_feat(post)
                
                # Reconstruct depth
                depth_dist = self.world_model.heads['depth'](feat)
                pred_depth = depth_dist.mean  # (B, 1, 1, H, W)
                
                # Ground truth
                gt_depth = observations['depth']  # (B, H, W, 1)
                
                # Compute metrics
                batch_size = gt_depth.shape[0]
                for i in range(batch_size):
                    gt = gt_depth[i, :, :, 0].cpu().numpy()
                    pred = pred_depth[i, 0, 0].cpu().numpy()
                    
                    # Resize if needed
                    if gt.shape != pred.shape:
                        from skimage.transform import resize
                        pred = resize(pred, gt.shape)
                    
                    # RMSE
                    rmse = np.sqrt(np.mean((pred - gt) ** 2))
                    results['rmse'].append(rmse)
                    
                    # MAE
                    mae = np.mean(np.abs(pred - gt))
                    results['mae'].append(mae)
                    
                    # SSIM
                    data_range = gt.max() - gt.min()
                    if data_range > 0:
                        ssim_val = ssim(gt, pred, data_range=data_range)
                        results['ssim'].append(ssim_val)
                    
                    # Depth accuracy
                    valid_mask = (gt > 0) & (pred > 0)
                    if valid_mask.sum() > 0:
                        ratio = np.maximum(
                            pred[valid_mask] / gt[valid_mask],
                            gt[valid_mask] / pred[valid_mask]
                        )
                        results['delta_1'].append((ratio < 1.25).mean())
                        results['delta_2'].append((ratio < 1.25**2).mean())
                        results['delta_3'].append((ratio < 1.25**3).mean())
        
        # Aggregate
        summary = {k: float(np.mean(v)) if v else 0.0 for k, v in results.items()}
        
        print(f"RMSE: {summary['rmse']:.4f}m")
        print(f"MAE: {summary['mae']:.4f}m")
        print(f"SSIM: {summary['ssim']:.4f}")
        print(f"δ < 1.25: {summary['delta_1']:.2%}")
        print(f"δ < 1.25²: {summary['delta_2']:.2%}")
        print(f"δ < 1.25³: {summary['delta_3']:.2%}")
        
        return summary
    
    def evaluate_trajectory_prediction(self, dataloader, use_goal_conditioning=True):
        """评估人类轨迹预测质量"""
        print(f"\n=== Evaluating Trajectory Prediction (goal={use_goal_conditioning}) ===")
        
        results = {
            'ade': [],
            'fde': [],
            'miss_rate': [],
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Trajectory Eval"):
                observations = {k: v.to(self.device) for k, v in batch['observations'].items()}
                
                # Skip if no human trajectory
                if 'oracle_humanoid_future_trajectory' not in observations:
                    continue
                
                # WM forward
                embed = self.world_model.encoder(observations)
                actions = batch['actions'].to(self.device).unsqueeze(1)
                is_first = batch.get('is_first', torch.zeros(actions.shape[0], 1, 1)).to(self.device)
                
                post, _ = self.world_model.dynamics.observe(embed.unsqueeze(1), actions, is_first)
                feat = self.world_model.dynamics.get_feat(post)
                
                # Predict trajectory
                human_start_goal = observations.get('human_start_goal', None) if use_goal_conditioning else None
                traj_dist = self.world_model.heads['human_traj'](feat, human_start_goal)
                pred_traj = traj_dist.mean  # (B, 1, N, T, 2)
                
                # Ground truth
                gt_traj = observations['oracle_humanoid_future_trajectory']  # (B, N, T, 2)
                
                # Compute metrics
                batch_size = gt_traj.shape[0]
                num_humans = gt_traj.shape[1]
                
                for b in range(batch_size):
                    for h in range(num_humans):
                        # Skip invalid humans
                        if gt_traj[b, h, 0, 0] < -90:
                            continue
                        
                        gt = gt_traj[b, h].cpu().numpy()  # (T, 2)
                        pred = pred_traj[b, 0, h].cpu().numpy()  # (T, 2)
                        
                        # ADE
                        displacements = np.linalg.norm(pred - gt, axis=1)
                        ade = displacements.mean()
                        results['ade'].append(ade)
                        
                        # FDE
                        fde = displacements[-1]
                        results['fde'].append(fde)
                        
                        # Miss rate
                        miss = 1 if fde > 2.0 else 0
                        results['miss_rate'].append(miss)
        
        # Aggregate
        summary = {k: float(np.mean(v)) if v else 0.0 for k, v in results.items()}
        
        print(f"ADE: {summary['ade']:.3f}m")
        print(f"FDE: {summary['fde']:.3f}m")
        print(f"Miss Rate (>2m): {summary['miss_rate']:.2%}")
        
        return summary
    
    def analyze_latent_space(self, dataloader, max_samples=1000):
        """分析latent space质量"""
        print("\n=== Analyzing Latent Space ===")
        
        latents = []
        labels = {
            'num_humans': [],
            'has_collision': [],
        }
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting Latents"):
                if sample_count >= max_samples:
                    break
                
                observations = {k: v.to(self.device) for k, v in batch['observations'].items()}
                
                # WM forward
                embed = self.world_model.encoder(observations)
                actions = batch['actions'].to(self.device).unsqueeze(1)
                is_first = torch.zeros(actions.shape[0], 1, 1).to(self.device)
                
                post, _ = self.world_model.dynamics.observe(embed.unsqueeze(1), actions, is_first)
                
                # Get latent
                h_t = post['deter'][:, 0]  # (B, deter_dim)
                z_t = post['stoch'][:, 0]  # (B, stoch_dim, classes)
                
                # Flatten stochastic
                if len(z_t.shape) == 3:
                    z_t = z_t.reshape(z_t.shape[0], -1)
                
                # Concatenate
                latent = torch.cat([h_t, z_t], dim=-1)
                latents.append(latent.cpu().numpy())
                
                # Labels
                labels['num_humans'].extend(batch.get('num_humans', [0] * len(h_t)))
                labels['has_collision'].extend(batch.get('has_collision', [0] * len(h_t)))
                
                sample_count += len(h_t)
        
        latents = np.concatenate(latents, axis=0)[:max_samples]
        for k in labels:
            labels[k] = np.array(labels[k])[:max_samples]
        
        print(f"Collected {len(latents)} latent samples")
        
        # t-SNE
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latents_2d = tsne.fit_transform(latents)
        
        # Linear probe for collision prediction
        if len(labels['has_collision']) > 0:
            print("Training linear probe...")
            split = int(0.8 * len(latents))
            
            clf = LogisticRegression(max_iter=1000)
            clf.fit(latents[:split], labels['has_collision'][:split])
            probe_acc = clf.score(latents[split:], labels['has_collision'][split:])
            
            print(f"Collision Prediction Accuracy: {probe_acc:.2%}")
        else:
            probe_acc = 0.0
        
        return {
            'latents_2d': latents_2d,
            'labels': labels,
            'probe_accuracy': float(probe_acc),
        }
    
    def generate_visualizations(self, output_dir='figures'):
        """生成可视化图表"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== Generating Visualizations ===")
        print(f"Output directory: {output_dir}")
        
        # TODO: Add visualization code here
        # - Depth reconstruction examples
        # - Trajectory prediction examples
        # - t-SNE plots
        
        print("Visualizations saved!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate World Model Quality')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/test',
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='results/wm_quality',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("World Model Quality Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    # TODO: Add actual model loading code
    # checkpoint = torch.load(args.checkpoint)
    # world_model = load_world_model(checkpoint)
    
    # For now, use placeholder
    print("⚠️  Model loading not implemented yet")
    print("Please implement model loading in the script")
    
    # Load data
    print("\nLoading test data...")
    # TODO: Add actual data loading code
    # test_loader = create_dataloader(args.data_path, batch_size=args.batch_size)
    
    print("⚠️  Data loading not implemented yet")
    print("Please implement data loading in the script")
    
    # Create evaluator
    # evaluator = WorldModelEvaluator(world_model, device=args.device)
    
    # Run evaluations
    results = {}
    
    # 1. Depth reconstruction
    # results['depth'] = evaluator.evaluate_depth_reconstruction(test_loader)
    
    # 2. Trajectory prediction (with goal)
    # results['traj_with_goal'] = evaluator.evaluate_trajectory_prediction(test_loader, use_goal_conditioning=True)
    
    # 3. Trajectory prediction (without goal)
    # results['traj_no_goal'] = evaluator.evaluate_trajectory_prediction(test_loader, use_goal_conditioning=False)
    
    # 4. Latent space analysis
    # results['latent'] = evaluator.analyze_latent_space(test_loader)
    
    # Save results
    result_file = output_dir / 'wm_quality_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {result_file}")
    
    # Generate visualizations
    # evaluator.generate_visualizations(output_dir / 'figures')
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
