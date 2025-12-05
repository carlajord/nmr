"""
Model Comparison and Visualization Tools

This script provides utilities to:
1. Compare baseline and advanced models side-by-side
2. Visualize training progress
3. Analyze prediction quality
4. Generate performance reports
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from src.baseline_model import BaselineCNN, NMRDataset, train_model, evaluate_model, run_all
from src.advanced_model import AdvancedNMRNet, train_advanced_model, evaluate_advanced_model
from torch.utils.data import DataLoader, TensorDataset
import time
from src.utils import verify_split_after_shuffle, set_seed, get_data_splits_normalized

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results")

def plot_training_history(
    baseline_history: Dict,
    advanced_history: Dict,
    save_path: str = os.path.join(RESULTS_PATH, 'training_comparison.png')
):
    """
    Plot and compare training histories of both models.
    
    Args:
        baseline_history: Training history from baseline model
        advanced_history: Training history from advanced model
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    axes[0].plot(baseline_history['train_loss'], label='Baseline Train', 
                 linewidth=2, alpha=0.8)
    axes[0].plot(baseline_history['val_loss'], label='Baseline Val', 
                 linewidth=2, alpha=0.8)
    axes[0].plot(advanced_history['train_loss'], label='Advanced Train', 
                 linewidth=2, alpha=0.8, linestyle='--')
    axes[0].plot(advanced_history['val_loss'], label='Advanced Val', 
                 linewidth=2, alpha=0.8, linestyle='--')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot validation loss only (zoomed in for better comparison)
    axes[1].plot(baseline_history['val_loss'], label='Baseline', 
                 linewidth=2, alpha=0.8, marker='o', markersize=3)
    axes[1].plot(advanced_history['val_loss'], label='Advanced', 
                 linewidth=2, alpha=0.8, marker='s', markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training comparison plot saved to: {save_path}")
    plt.close()


def plot_prediction_analysis(
    baseline_preds: np.ndarray,
    advanced_preds: np.ndarray,
    targets: np.ndarray,
    save_path: str = os.path.join(RESULTS_PATH, 'prediction_analysis.png')
):
    """
    Create comprehensive prediction analysis plots.
    
    Args:
        baseline_preds: Predictions from baseline model
        advanced_preds: Predictions from advanced model
        targets: Ground truth values
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Scatter plots: Predicted vs Actual
    axes[0, 0].scatter(targets, baseline_preds, alpha=0.5, s=20)
    axes[0, 0].plot([targets.min(), targets.max()], 
                    [targets.min(), targets.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual MQI', fontsize=11)
    axes[0, 0].set_ylabel('Predicted MQI', fontsize=11)
    axes[0, 0].set_title('Baseline Model: Predicted vs Actual', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(targets, advanced_preds, alpha=0.5, s=20, color='orange')
    axes[0, 1].plot([targets.min(), targets.max()], 
                    [targets.min(), targets.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel('Actual MQI', fontsize=11)
    axes[0, 1].set_ylabel('Predicted MQI', fontsize=11)
    axes[0, 1].set_title('Advanced Model: Predicted vs Actual', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution
    baseline_errors = baseline_preds.flatten() - targets.flatten()
    advanced_errors = advanced_preds.flatten() - targets.flatten()
    
    axes[0, 2].hist(baseline_errors, bins=50, alpha=0.6, label='Baseline', 
                    color='blue', edgecolor='black')
    axes[0, 2].hist(advanced_errors, bins=50, alpha=0.6, label='Advanced', 
                    color='orange', edgecolor='black')
    axes[0, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 2].set_xlabel('Prediction Error', fontsize=11)
    axes[0, 2].set_ylabel('Frequency', fontsize=11)
    axes[0, 2].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Residual plots
    axes[1, 0].scatter(targets, baseline_errors, alpha=0.5, s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Actual MQI', fontsize=11)
    axes[1, 0].set_ylabel('Residual', fontsize=11)
    axes[1, 0].set_title('Baseline Model: Residual Plot', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(targets, advanced_errors, alpha=0.5, s=20, color='orange')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Actual MQI', fontsize=11)
    axes[1, 1].set_ylabel('Residual', fontsize=11)
    axes[1, 1].set_title('Advanced Model: Residual Plot', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Absolute error comparison
    baseline_abs_errors = np.abs(baseline_errors)
    advanced_abs_errors = np.abs(advanced_errors)
    
    comparison_data = [baseline_abs_errors, advanced_abs_errors]
    bp = axes[1, 2].boxplot(comparison_data, labels=['Baseline', 'Advanced'],
                            patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    axes[1, 2].set_ylabel('Absolute Error', fontsize=11)
    axes[1, 2].set_title('Absolute Error Distribution', 
                         fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction analysis plot saved to: {save_path}")
    plt.close()


def plot_baseline_models_comparison(
    xgboost_preds: np.ndarray,
    gp_preds: np.ndarray,
    minimal_cnn_preds: np.ndarray,
    targets: np.ndarray,
    save_path: str = os.path.join(RESULTS_PATH, 'baseline_models_comparison.png')
):
    """
    Create comprehensive comparison plots for the three baseline models.

    Args:
        xgboost_preds: Predictions from XGBoost model
        gp_preds: Predictions from Gaussian Process model
        minimal_cnn_preds: Predictions from Minimal CNN model
        targets: Ground truth values
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Colors for each model
    colors = {
        'xgboost': '#1f77b4',      # Blue
        'gp': '#ff7f0e',            # Orange
        'minimal_cnn': '#2ca02c'    # Green
    }

    # ========== ROW 1: SCATTER PLOTS (Predicted vs Actual) ==========

    # XGBoost scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(targets, xgboost_preds, alpha=0.5, s=20, color=colors['xgboost'])
    ax1.plot([targets.min(), targets.max()],
             [targets.min(), targets.max()],
             'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual MQI', fontsize=11)
    ax1.set_ylabel('Predicted MQI', fontsize=11)
    ax1.set_title('XGBoost: Predicted vs Actual', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gaussian Process scatter plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(targets, gp_preds, alpha=0.5, s=20, color=colors['gp'])
    ax2.plot([targets.min(), targets.max()],
             [targets.min(), targets.max()],
             'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual MQI', fontsize=11)
    ax2.set_ylabel('Predicted MQI', fontsize=11)
    ax2.set_title('Gaussian Process: Predicted vs Actual', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Minimal CNN scatter plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(targets, minimal_cnn_preds, alpha=0.5, s=20, color=colors['minimal_cnn'])
    ax3.plot([targets.min(), targets.max()],
             [targets.min(), targets.max()],
             'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual MQI', fontsize=11)
    ax3.set_ylabel('Predicted MQI', fontsize=11)
    ax3.set_title('Minimal CNN: Predicted vs Actual', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Combined scatter plot
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(targets, xgboost_preds, alpha=0.4, s=15, color=colors['xgboost'], label='XGBoost')
    ax4.scatter(targets, gp_preds, alpha=0.4, s=15, color=colors['gp'], label='GP')
    ax4.scatter(targets, minimal_cnn_preds, alpha=0.4, s=15, color=colors['minimal_cnn'], label='Minimal CNN')
    ax4.plot([targets.min(), targets.max()],
             [targets.min(), targets.max()],
             'r--', linewidth=2, label='Perfect')
    ax4.set_xlabel('Actual MQI', fontsize=11)
    ax4.set_ylabel('Predicted MQI', fontsize=11)
    ax4.set_title('All Models Combined', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ========== ROW 2: RESIDUAL PLOTS ==========

    xgboost_errors = xgboost_preds.flatten() - targets.flatten()
    gp_errors = gp_preds.flatten() - targets.flatten()
    minimal_cnn_errors = minimal_cnn_preds.flatten() - targets.flatten()

    # XGBoost residuals
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.scatter(targets, xgboost_errors, alpha=0.5, s=20, color=colors['xgboost'])
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Actual MQI', fontsize=11)
    ax5.set_ylabel('Residual', fontsize=11)
    ax5.set_title('XGBoost: Residual Plot', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Gaussian Process residuals
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.scatter(targets, gp_errors, alpha=0.5, s=20, color=colors['gp'])
    ax6.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax6.set_xlabel('Actual MQI', fontsize=11)
    ax6.set_ylabel('Residual', fontsize=11)
    ax6.set_title('GP: Residual Plot', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Minimal CNN residuals
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.scatter(targets, minimal_cnn_errors, alpha=0.5, s=20, color=colors['minimal_cnn'])
    ax7.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax7.set_xlabel('Actual MQI', fontsize=11)
    ax7.set_ylabel('Residual', fontsize=11)
    ax7.set_title('Minimal CNN: Residual Plot', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Absolute error boxplot comparison
    ax8 = fig.add_subplot(gs[1, 3])
    xgboost_abs_errors = np.abs(xgboost_errors)
    gp_abs_errors = np.abs(gp_errors)
    minimal_cnn_abs_errors = np.abs(minimal_cnn_errors)

    comparison_data = [xgboost_abs_errors, gp_abs_errors, minimal_cnn_abs_errors]
    bp = ax8.boxplot(comparison_data, labels=['XGBoost', 'GP', 'Min-CNN'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor(colors['xgboost'])
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(colors['gp'])
    bp['boxes'][1].set_alpha(0.6)
    bp['boxes'][2].set_facecolor(colors['minimal_cnn'])
    bp['boxes'][2].set_alpha(0.6)
    ax8.set_ylabel('Absolute Error', fontsize=11)
    ax8.set_title('Absolute Error Comparison', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')

    # ========== ROW 3: ERROR DISTRIBUTIONS ==========

    # XGBoost error distribution
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.hist(xgboost_errors, bins=40, alpha=0.7, color=colors['xgboost'], edgecolor='black')
    ax9.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax9.axvline(x=np.mean(xgboost_errors), color='darkblue', linestyle=':', linewidth=2,
                label=f'Mean: {np.mean(xgboost_errors):.4f}')
    ax9.set_xlabel('Prediction Error', fontsize=11)
    ax9.set_ylabel('Frequency', fontsize=11)
    ax9.set_title('XGBoost: Error Distribution', fontsize=12, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # GP error distribution
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.hist(gp_errors, bins=40, alpha=0.7, color=colors['gp'], edgecolor='black')
    ax10.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax10.axvline(x=np.mean(gp_errors), color='darkorange', linestyle=':', linewidth=2,
                 label=f'Mean: {np.mean(gp_errors):.4f}')
    ax10.set_xlabel('Prediction Error', fontsize=11)
    ax10.set_ylabel('Frequency', fontsize=11)
    ax10.set_title('GP: Error Distribution', fontsize=12, fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # Minimal CNN error distribution
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.hist(minimal_cnn_errors, bins=40, alpha=0.7, color=colors['minimal_cnn'], edgecolor='black')
    ax11.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax11.axvline(x=np.mean(minimal_cnn_errors), color='darkgreen', linestyle=':', linewidth=2,
                 label=f'Mean: {np.mean(minimal_cnn_errors):.4f}')
    ax11.set_xlabel('Prediction Error', fontsize=11)
    ax11.set_ylabel('Frequency', fontsize=11)
    ax11.set_title('Minimal CNN: Error Distribution', fontsize=12, fontweight='bold')
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # Combined error distribution
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.hist(xgboost_errors, bins=30, alpha=0.5, color=colors['xgboost'],
              edgecolor='black', label='XGBoost')
    ax12.hist(gp_errors, bins=30, alpha=0.5, color=colors['gp'],
              edgecolor='black', label='GP')
    ax12.hist(minimal_cnn_errors, bins=30, alpha=0.5, color=colors['minimal_cnn'],
              edgecolor='black', label='Minimal CNN')
    ax12.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax12.set_xlabel('Prediction Error', fontsize=11)
    ax12.set_ylabel('Frequency', fontsize=11)
    ax12.set_title('Combined Error Distribution', fontsize=12, fontweight='bold')
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Baseline models comparison plot saved to: {save_path}")
    plt.close()


def generate_comparison_report(
    baseline_metrics: Dict,
    advanced_metrics: Dict,
    baseline_time: float,
    advanced_time: float,
    save_path: str = os.path.join(RESULTS_PATH, 'comparison_report.txt')
):
    """
    Generate a detailed text report comparing both models.
    
    Args:
        baseline_metrics: Dictionary with baseline model metrics
        advanced_metrics: Dictionary with advanced model metrics
        baseline_time: Training time for baseline model (seconds)
        advanced_time: Training time for advanced model (seconds)
        save_path: Path to save the report
    """
    report = []
    report.append("=" * 80)
    report.append("NMR MQI PREDICTION - MODEL COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Baseline model section
    report.append("BASELINE MODEL (Simple CNN)")
    report.append("-" * 80)
    report.append(f"MSE:              {baseline_metrics['mse']:.6f}")
    report.append(f"MAE:              {baseline_metrics['mae']:.6f}")
    report.append(f"RMSE:             {baseline_metrics['rmse']:.6f}")
    report.append(f"Training Time:    {baseline_time:.2f} seconds ({baseline_time/60:.2f} minutes)")
    report.append(f"Parameters:       {baseline_metrics['params']:,}")
    report.append("")
    
    # Advanced model section
    report.append("ADVANCED MODEL (Attention + Residual + Multi-scale)")
    report.append("-" * 80)
    report.append(f"MSE:              {advanced_metrics['mse']:.6f}")
    report.append(f"MAE:              {advanced_metrics['mae']:.6f}")
    report.append(f"RMSE:             {advanced_metrics['rmse']:.6f}")
    report.append(f"R² Score:         {advanced_metrics['r2']:.6f}")
    report.append(f"Training Time:    {advanced_time:.2f} seconds ({advanced_time/60:.2f} minutes)")
    report.append(f"Parameters:       {advanced_metrics['params']:,}")
    report.append("")
    
    # Improvement analysis
    report.append("IMPROVEMENT ANALYSIS")
    report.append("-" * 80)
    mse_improvement = ((baseline_metrics['mse'] - advanced_metrics['mse']) / 
                       baseline_metrics['mse'] * 100)
    mae_improvement = ((baseline_metrics['mae'] - advanced_metrics['mae']) / 
                       baseline_metrics['mae'] * 100)
    rmse_improvement = ((baseline_metrics['rmse'] - advanced_metrics['rmse']) / 
                        baseline_metrics['rmse'] * 100)
    
    report.append(f"MSE Improvement:  {mse_improvement:+.2f}%")
    report.append(f"MAE Improvement:  {mae_improvement:+.2f}%")
    report.append(f"RMSE Improvement: {rmse_improvement:+.2f}%")
    report.append(f"Extra Time:       {advanced_time - baseline_time:.2f} seconds")
    report.append(f"Parameter Ratio:  {advanced_metrics['params'] / baseline_metrics['params']:.2f}x")
    report.append("")
    
    # Key insights
    report.append("KEY INSIGHTS")
    report.append("-" * 80)
    if mse_improvement > 0:
        report.append(f"✓ Advanced model shows {mse_improvement:.1f}% better MSE performance")
    else:
        report.append(f"✗ Advanced model shows {abs(mse_improvement):.1f}% worse MSE performance")
    
    if mae_improvement > 0:
        report.append(f"✓ Advanced model shows {mae_improvement:.1f}% better MAE performance")
    else:
        report.append(f"✗ Advanced model shows {abs(mae_improvement):.1f}% worse MAE performance")
    
    param_increase = (advanced_metrics['params'] - baseline_metrics['params']) / baseline_metrics['params'] * 100
    report.append(f"• Advanced model uses {param_increase:.1f}% more parameters")
    
    time_increase = (advanced_time - baseline_time) / baseline_time * 100
    report.append(f"• Advanced model takes {time_increase:.1f}% longer to train")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 80)
    if mse_improvement > 5:
        report.append("→ Use Advanced Model: Significant improvement justifies extra complexity")
    elif mse_improvement > 0:
        report.append("→ Consider Advanced Model: Modest improvement, evaluate based on requirements")
    else:
        report.append("→ Use Baseline Model: Simpler model performs adequately")
    report.append("")
    
    report.append("=" * 80)
    
    # Write report to file
    report_text = "\n".join(report)
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {save_path}")


def run_complete_comparison(
    batch_size: int = 32,
    baseline_epochs: int = 50,
    advanced_epochs: int = 100,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
):
    """
    Run complete comparison between baseline and advanced models.

    This function:
    1. Sets random seeds for reproducibility
    2. Loads and splits data with proper normalization (NO LEAKAGE)
    3. Trains both models
    4. Evaluates both models
    5. Generates visualizations and reports

    Args:
        batch_size: Batch size for training (default: 32)
        baseline_epochs: Number of epochs for baseline model (default: 50)
        advanced_epochs: Number of epochs for advanced model (default: 100)
        seed: Random seed for reproducibility (default: 42)
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
    """
    print("=" * 80)
    print("STARTING COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    print()

    # Step 1: Set all random seeds for reproducibility
    set_seed(seed)

    # Step 2: Load data with proper normalization (prevents leakage!)
    train_data, val_data, test_data, norm_stats = get_data_splits_normalized(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )

    # Unpack data
    echo_train, mqi_train = train_data
    echo_val, mqi_val = val_data
    echo_test, mqi_test = test_data

    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(echo_train),
        torch.FloatTensor(mqi_train).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(echo_val),
        torch.FloatTensor(mqi_val).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(echo_test),
        torch.FloatTensor(mqi_test).unsqueeze(1)
    )

    # Verify split quality
    verify_split_after_shuffle(train_dataset, val_dataset, test_dataset)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataset Information:")
    print(f"  Training samples:   {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples:       {len(test_dataset)}")
    print(f"  Batch size:         {batch_size}")
    
    # ========== BASELINE MODEL ==========
    print("\n" + "=" * 80)
    print("TRAINING BASELINE MODEL")
    print("=" * 80)
    
    baseline_model = BaselineCNN(dropout_rate=0.1)
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"Baseline model parameters: {baseline_params:,}")
    
    baseline_start = time.time()
    baseline_history = train_model(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=baseline_epochs,
        learning_rate=0.0001
    )
    baseline_time = time.time() - baseline_start
    
    # Load best baseline model and evaluate
    baseline_model.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_baseline_model.pth')))
    baseline_mse, baseline_mae, baseline_preds, baseline_targets = evaluate_model(
        baseline_model, test_loader
    )
    
    # ========== ADVANCED MODEL ==========
    print("\n" + "=" * 80)
    print("TRAINING ADVANCED MODEL")
    print("=" * 80)
    
    advanced_model = AdvancedNMRNet(dropout_rate=0.1)
    advanced_params = sum(p.numel() for p in advanced_model.parameters())
    print(f"Advanced model parameters: {advanced_params:,}")
    
    advanced_start = time.time()
    advanced_history = train_advanced_model(
        model=advanced_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=advanced_epochs,
        learning_rate=0.0001,
        use_early_stopping=True
    )
    advanced_time = time.time() - advanced_start
    
    # Load best advanced model and evaluate
    advanced_model.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_advanced_model.pth')))
    advanced_mse, advanced_mae, advanced_r2, advanced_preds, advanced_targets = \
        evaluate_advanced_model(advanced_model, test_loader)
    
    # ========== COMPARISON AND VISUALIZATION ==========
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON REPORT AND VISUALIZATIONS")
    print("=" * 80)
    
    # Prepare metrics dictionaries
    baseline_metrics = {
        'mse': baseline_mse,
        'mae': baseline_mae,
        'rmse': np.sqrt(baseline_mse),
        'params': baseline_params
    }
    
    advanced_metrics = {
        'mse': advanced_mse,
        'mae': advanced_mae,
        'rmse': np.sqrt(advanced_mse),
        'r2': advanced_r2,
        'params': advanced_params
    }
    
    # Generate visualizations
    plot_training_history(baseline_history, advanced_history)
    plot_prediction_analysis(baseline_preds, advanced_preds, baseline_targets)
    
    # Generate comparison report
    generate_comparison_report(
        baseline_metrics, advanced_metrics,
        baseline_time, advanced_time
    )
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  1. {os.path.join(RESULTS_PATH, 'best_baseline_model.pth')}")
    print(f"  2. {os.path.join(RESULTS_PATH, 'best_advanced_model.pth')}")
    print(f"  3. {os.path.join(RESULTS_PATH, 'training_comparison.png')}")
    print(f"  4. {os.path.join(RESULTS_PATH, 'prediction_analysis.png')}")
    print(f"  5. {os.path.join(RESULTS_PATH, 'comparison_report.txt')}")



def run_baseline_comparison():
    """
    Run comparison among the three baseline models: XGBoost, Gaussian Process, and Minimal CNN.
    This function assumes that the predictions from each model are already available.
    """
    # Load test targets and predictions from each baseline model
    # (Assuming these are saved as .npy files in the results directory)
    xgb_preds, gp_preds, min_preds, y_test, \
               xgboost_metrics, gp_metrics, minimal_cnn_metrics = run_all()

    # Generate visualization
    plot_baseline_models_comparison(
        xgboost_preds=xgb_preds,
        gp_preds=gp_preds,
        minimal_cnn_preds=min_preds.flatten(),
        targets=y_test
    )

    print("\n" + "="*80)
    print("BASELINE MODELS COMPARISON COMPLETE!")
print("="*80)
print("\nGenerated files:")
print(f"  1. {os.path.join(RESULTS_PATH, 'baseline_models_comparison.png')}")
print(f"  2. {os.path.join(RESULTS_PATH, 'baseline_models_report.txt')}")     

# Example usage
if __name__ == "__main__":
    # Run complete comparison
    # Data loading, shuffling, splitting, and normalization are all handled internally
    # with proper seed setting and no data leakage!
    run_complete_comparison(
        batch_size=32,
        baseline_epochs=50,
        advanced_epochs=100,
        seed=42,  # For reproducibility
        train_ratio=0.7,
        val_ratio=0.15
    )
