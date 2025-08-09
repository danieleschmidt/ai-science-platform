"""Advanced visualization utilities for research dashboards"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict, Any
import seaborn as sns


def plot_model_performance(model_metrics, 
                          save_path: Optional[str] = None,
                          figsize: tuple = (14, 10)) -> plt.Figure:
    """Plot comprehensive model performance analysis"""
    if not isinstance(model_metrics, list):
        model_metrics = [model_metrics]
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Extract metrics data
    accuracies = [m.accuracy for m in model_metrics]
    losses = [m.loss for m in model_metrics]
    train_times = [m.training_time for m in model_metrics]
    inference_times = [m.inference_time for m in model_metrics]
    
    # Accuracy over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(len(accuracies)), accuracies, 'o-', color='green', linewidth=2)
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Validation Run')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # Loss over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(len(losses)), losses, 'o-', color='red', linewidth=2)
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Validation Run')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Training vs Inference time
    ax3 = fig.add_subplot(gs[0, 2])
    x_pos = np.arange(2)
    times = [np.mean(train_times), np.mean(inference_times)]
    bars = ax3.bar(x_pos, times, color=['blue', 'orange'], alpha=0.7)
    ax3.set_title('Training vs Inference Time')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['Training', 'Inference'])
    ax3.set_ylabel('Time (seconds)')
    
    # Performance scatter plot
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(losses, accuracies, c=train_times, cmap='viridis', s=50, alpha=0.7)
    ax4.set_xlabel('Loss')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy vs Loss')
    
    # Additional metrics heatmap
    if model_metrics and model_metrics[0].additional_metrics:
        ax5 = fig.add_subplot(gs[1, 1:])
        
        metric_names = list(model_metrics[0].additional_metrics.keys())
        metric_values = []
        for m in model_metrics:
            metric_values.append([m.additional_metrics[name] for name in metric_names])
        
        metric_values = np.array(metric_values).T
        
        im = ax5.imshow(metric_values, cmap='RdYlBu_r', aspect='auto')
        ax5.set_xticks(range(len(model_metrics)))
        ax5.set_yticks(range(len(metric_names)))
        ax5.set_xticklabels([f'Run {i+1}' for i in range(len(model_metrics))])
        ax5.set_yticklabels(metric_names)
        ax5.set_title('Additional Metrics Heatmap')
        
        plt.colorbar(im, ax=ax5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cross_validation_results(cv_metrics: List, 
                                 save_path: Optional[str] = None,
                                 figsize: tuple = (12, 8)) -> plt.Figure:
    """Plot cross-validation results analysis"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Extract data
    accuracies = [m.accuracy for m in cv_metrics]
    losses = [m.loss for m in cv_metrics]
    folds = range(1, len(cv_metrics) + 1)
    
    # Accuracy across folds
    axes[0].bar(folds, accuracies, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0].axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(accuracies):.3f}')
    axes[0].set_title('Accuracy Across CV Folds')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss across folds
    axes[1].bar(folds, losses, color='lightcoral', alpha=0.7, edgecolor='black')
    axes[1].axhline(y=np.mean(losses), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(losses):.3f}')
    axes[1].set_title('Loss Across CV Folds')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Performance distribution
    axes[2].hist(accuracies, bins=min(5, len(accuracies)), alpha=0.7, 
                 color='skyblue', edgecolor='black')
    axes[2].axvline(x=np.mean(accuracies), color='blue', linestyle='--')
    axes[2].set_title('Accuracy Distribution')
    axes[2].set_xlabel('Accuracy')
    axes[2].set_ylabel('Frequency')
    
    # Stability analysis
    axes[3].plot(folds, accuracies, 'o-', color='green', label='Accuracy', linewidth=2)
    axes[3].fill_between(folds, 
                         np.mean(accuracies) - np.std(accuracies), 
                         np.mean(accuracies) + np.std(accuracies), 
                         alpha=0.3, color='green')
    axes[3].set_title('Model Stability')
    axes[3].set_xlabel('Fold')
    axes[3].set_ylabel('Score')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_research_dashboard(discoveries, model_metrics=None, experiment_results=None,
                             save_path: Optional[str] = None,
                             figsize: tuple = (20, 15)) -> plt.Figure:
    """Create comprehensive research dashboard"""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('AI Science Platform - Research Dashboard', fontsize=20, y=0.95)
    
    # Discovery confidence scores
    ax1 = fig.add_subplot(gs[0, :2])
    if discoveries:
        confidences = [d.confidence for d in discoveries]
        ax1.bar(range(len(discoveries)), confidences, 
                color=plt.cm.viridis([c for c in confidences]))
        ax1.set_title('Discovery Confidence Scores')
        ax1.set_xlabel('Discovery Index')
        ax1.set_ylabel('Confidence')
        ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
    else:
        ax1.text(0.5, 0.5, 'No discoveries yet', ha='center', va='center', 
                transform=ax1.transAxes)
        ax1.set_title('Discovery Results')
    
    # Model performance
    ax2 = fig.add_subplot(gs[0, 2:])
    if isinstance(model_metrics, list) and model_metrics:
        accuracies = [m.accuracy for m in model_metrics]
        losses = [m.loss for m in model_metrics]
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(range(len(accuracies)), accuracies, 'g-o', label='Accuracy')
        line2 = ax2_twin.plot(range(len(losses)), losses, 'r-s', label='Loss')
        ax2.set_xlabel('Validation Run')
        ax2.set_ylabel('Accuracy', color='g')
        ax2_twin.set_ylabel('Loss', color='r')
        ax2.set_title('Model Performance')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='center right')
    else:
        ax2.text(0.5, 0.5, 'No model metrics available', ha='center', va='center',
                transform=ax2.transAxes)
        ax2.set_title('Model Performance')
    
    # Experiment comparison
    ax3 = fig.add_subplot(gs[1, :2])
    if experiment_results and 'experiments' in experiment_results:
        exp_names = []
        exp_means = []
        for name, data in experiment_results['experiments'].items():
            if 'error' not in data:
                exp_names.append(name[:15] + '...' if len(name) > 15 else name)
                exp_means.append(data.get('mean', 0))
        
        if exp_names:
            ax3.bar(range(len(exp_names)), exp_means, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(exp_names))))
            ax3.set_xticks(range(len(exp_names)))
            ax3.set_xticklabels(exp_names, rotation=45, ha='right')
            ax3.set_title(f'Experiment Results - {experiment_results.get("metric", "Score")}')
            ax3.set_ylabel('Score')
        else:
            ax3.text(0.5, 0.5, 'No valid experiment data', ha='center', va='center',
                    transform=ax3.transAxes)
            ax3.set_title('Experiment Results')
    else:
        ax3.text(0.5, 0.5, 'No experiment results', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title('Experiment Results')
    
    # Discovery distribution
    ax4 = fig.add_subplot(gs[1, 2:])
    if discoveries:
        confidences = [d.confidence for d in discoveries]
        ax4.hist(confidences, bins=min(10, len(discoveries)), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(x=np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax4.set_title('Discovery Confidence Distribution')
        ax4.set_xlabel('Confidence Score')
        ax4.set_ylabel('Count')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No discovery distribution data', ha='center', va='center',
                transform=ax4.transAxes)
        ax4.set_title('Discovery Distribution')
    
    # Summary statistics
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    summary_text = []
    summary_text.append("RESEARCH SUMMARY")
    summary_text.append("=" * 50)
    summary_text.append(f"Total Discoveries: {len(discoveries) if discoveries else 0}")
    
    if discoveries:
        confidences = [d.confidence for d in discoveries]
        summary_text.append(f"Average Confidence: {np.mean(confidences):.3f}")
        summary_text.append(f"Max Confidence: {np.max(confidences):.3f}")
    
    if isinstance(model_metrics, list) and model_metrics:
        accuracies = [m.accuracy for m in model_metrics]
        summary_text.append(f"Model Evaluations: {len(model_metrics)}")
        summary_text.append(f"Best Accuracy: {np.max(accuracies):.3f}")
        summary_text.append(f"Avg Training Time: {np.mean([m.training_time for m in model_metrics]):.3f}s")
    
    if experiment_results and 'experiments' in experiment_results:
        valid_exps = [name for name, data in experiment_results['experiments'].items() 
                     if 'error' not in data]
        summary_text.append(f"Successful Experiments: {len(valid_exps)}")
    
    ax5.text(0.05, 0.95, '\n'.join(summary_text), transform=ax5.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig