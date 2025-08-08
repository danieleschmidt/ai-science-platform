"""Visualization utilities for discovery results"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict, Any
import seaborn as sns


def plot_discovery_results(discoveries, 
                          save_path: Optional[str] = None,
                          figsize: tuple = (12, 8)) -> plt.Figure:
    """Plot discovery results and confidence scores
    
    Args:
        discoveries: List of Discovery objects
        save_path: Path to save the plot
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    if not discoveries:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No discoveries to plot', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Discovery Results')
        return fig
    
    # Extract data for plotting
    confidences = [d.confidence for d in discoveries]
    hypotheses = [d.hypothesis[:50] + '...' if len(d.hypothesis) > 50 
                 else d.hypothesis for d in discoveries]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Confidence scores
    bars = ax1.bar(range(len(discoveries)), confidences, 
                   color=plt.cm.viridis([c for c in confidences]))
    ax1.set_xlabel('Discovery Index')
    ax1.set_ylabel('Confidence Score')
    ax1.set_title('Discovery Confidence Scores')
    ax1.set_ylim(0, 1)
    
    # Add confidence threshold line
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, 
                label='Default Threshold (0.7)')
    ax1.legend()
    
    # Plot 2: Confidence distribution
    ax2.hist(confidences, bins=min(10, len(discoveries)), 
             alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Number of Discoveries')
    ax2.set_title('Distribution of Discovery Confidence')
    ax2.axvline(x=np.mean(confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidences):.3f}')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_experiment_comparison(results_dict: Dict[str, Any], 
                              metric: str,
                              save_path: Optional[str] = None,
                              figsize: tuple = (10, 6)) -> plt.Figure:
    """Compare experiment results across different configurations
    
    Args:
        results_dict: Dictionary with experiment results
        metric: Metric to compare
        save_path: Path to save the plot
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    experiment_names = []
    means = []
    stds = []
    
    for exp_name, exp_data in results_dict.get('experiments', {}).items():
        if 'error' not in exp_data:
            experiment_names.append(exp_name)
            means.append(exp_data['mean'])
            stds.append(exp_data['std'])
    
    if not experiment_names:
        ax.text(0.5, 0.5, f'No valid data for metric: {metric}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Experiment Comparison - {metric}')
        return fig
    
    # Create bar plot with error bars
    x_pos = np.arange(len(experiment_names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                  color=plt.cm.Set3(np.linspace(0, 1, len(experiment_names))))
    
    ax.set_xlabel('Experiments')
    ax.set_ylabel(f'{metric}')
    ax.set_title(f'Experiment Comparison - {metric}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(experiment_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.01 * max(means), 
                f'{mean:.3f}±{std:.3f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_data_distribution(data: np.ndarray, 
                          targets: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None,
                          figsize: tuple = (12, 8)) -> plt.Figure:
    """Plot data distribution and relationships
    
    Args:
        data: Input data array
        targets: Target array (optional)
        save_path: Path to save the plot
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    n_features = data.shape[1]
    
    if targets is not None:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Feature distribution
        axes[0].hist(data[:, 0], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('Feature Distribution')
        axes[0].set_xlabel('Feature Value')
        axes[0].set_ylabel('Frequency')
        
        # Target distribution
        axes[1].hist(targets, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].set_title('Target Distribution')
        axes[1].set_xlabel('Target Value')
        axes[1].set_ylabel('Frequency')
        
        # Scatter plot
        axes[2].scatter(data[:, 0], targets, alpha=0.6, s=20)
        axes[2].set_title('Feature vs Target')
        axes[2].set_xlabel('Feature Value')
        axes[2].set_ylabel('Target Value')
        
        # Correlation plot if multiple features
        if n_features > 1:
            corr_data = np.column_stack([data, targets])
            corr_matrix = np.corrcoef(corr_data.T)
            im = axes[3].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[3].set_title('Correlation Matrix')
            plt.colorbar(im, ax=axes[3])
        else:
            axes[3].text(0.5, 0.5, 'Single feature\ndata', 
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Feature Analysis')
        
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Feature distribution
        axes[0].hist(data[:, 0], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('Data Distribution')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        
        # Box plot
        axes[1].boxplot(data[:, 0])
        axes[1].set_title('Data Box Plot')
        axes[1].set_ylabel('Value')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig