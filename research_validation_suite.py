#!/usr/bin/env python3
"""
Comprehensive Research Validation Suite
Publication-ready validation and reproducibility framework for bioneural olfactory fusion research
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
import json
from pathlib import Path
import argparse

# Import our research components
from src.algorithms.bioneural_pipeline import BioneuralOlfactoryPipeline, PipelineConfig
from src.benchmarks.comparative_analysis import ComparativeAnalyzer
from src.benchmarks.statistical_validation import StatisticalValidator
from src.benchmarks.baseline_models import BaselineModelSuite

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResearchValidationSuite:
    """
    Comprehensive Research Validation Suite for Academic Publication
    
    Provides end-to-end validation framework for bioneural olfactory fusion research:
    1. Experimental design and data generation
    2. Comparative analysis against baseline methods
    3. Statistical significance testing with multiple comparison corrections
    4. Reproducibility validation across multiple runs
    5. Performance benchmarking and scalability analysis
    6. Publication-ready result compilation and visualization
    """
    
    def __init__(self, 
                 output_dir: str = "research_results",
                 random_seed: int = 42,
                 validation_config: Optional[Dict[str, Any]] = None):
        """
        Initialize research validation suite
        
        Args:
            output_dir: Directory for saving results and plots
            random_seed: Random seed for reproducibility
            validation_config: Configuration for validation parameters
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Default validation configuration
        default_config = {
            "signal_dimensions": [64, 128, 256],
            "dataset_sizes": [50, 100, 200],
            "num_validation_runs": 5,
            "statistical_alpha": 0.05,
            "effect_size_threshold": 0.5,
            "power_threshold": 0.8
        }
        
        self.config = {**default_config, **(validation_config or {})}
        
        # Initialize components
        self.pipeline = BioneuralOlfactoryPipeline()
        self.comparative_analyzer = ComparativeAnalyzer()
        self.statistical_validator = StatisticalValidator(
            alpha=self.config["statistical_alpha"],
            effect_size_threshold=self.config["effect_size_threshold"],
            power_threshold=self.config["power_threshold"]
        )
        
        # Results storage
        self.validation_results = {}
        
        logger.info(f"Research validation suite initialized: output_dir={output_dir}")
    
    def generate_synthetic_datasets(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate synthetic olfactory signal datasets for validation
        
        Returns:
            Dictionary of datasets with different characteristics
        """
        logger.info("Generating synthetic datasets for validation")
        
        datasets = {}
        
        for signal_dim in self.config["signal_dimensions"]:
            for dataset_size in self.config["dataset_sizes"]:
                dataset_name = f"synthetic_{signal_dim}D_{dataset_size}samples"
                
                # Generate diverse synthetic olfactory signals
                signals = []
                
                for i in range(dataset_size):
                    # Create diverse signal types
                    signal_type = i % 4
                    
                    if signal_type == 0:
                        # Gaussian mixture (complex molecular blend)
                        n_components = np.random.randint(2, 5)
                        signal = np.zeros(signal_dim)
                        for _ in range(n_components):
                            center = np.random.randint(0, signal_dim)
                            width = np.random.uniform(5, 20)
                            amplitude = np.random.uniform(0.5, 2.0)
                            gaussian = amplitude * np.exp(-((np.arange(signal_dim) - center) ** 2) / (2 * width ** 2))
                            signal += gaussian
                    
                    elif signal_type == 1:
                        # Exponential decay (volatile compounds)
                        decay_rates = np.random.uniform(0.01, 0.2, 3)
                        signal = np.zeros(signal_dim)
                        for rate in decay_rates:
                            component = np.exp(-rate * np.arange(signal_dim))
                            signal += component * np.random.uniform(0.3, 1.5)
                    
                    elif signal_type == 2:
                        # Oscillatory patterns (structured molecules)
                        frequencies = np.random.uniform(0.1, 2.0, 2)
                        phases = np.random.uniform(0, 2*np.pi, 2)
                        signal = np.zeros(signal_dim)
                        for freq, phase in zip(frequencies, phases):
                            oscillation = np.sin(freq * np.arange(signal_dim) + phase)
                            signal += oscillation * np.random.uniform(0.5, 1.0)
                    
                    else:
                        # Sparse signals (distinct molecular markers)
                        signal = np.zeros(signal_dim)
                        n_spikes = np.random.randint(3, 8)
                        spike_locations = np.random.choice(signal_dim, n_spikes, replace=False)
                        for loc in spike_locations:
                            signal[loc] = np.random.uniform(1.0, 3.0)
                    
                    # Add noise
                    noise_level = np.random.uniform(0.05, 0.2)
                    signal += np.random.normal(0, noise_level, signal_dim)
                    
                    # Normalize
                    signal = signal / (np.linalg.norm(signal) + 1e-10)
                    
                    signals.append(signal)
                
                # Split into training and test sets
                split_idx = int(0.7 * len(signals))
                datasets[dataset_name] = {
                    "training": np.array(signals[:split_idx]),
                    "test": np.array(signals[split_idx:]),
                    "metadata": {
                        "signal_dim": signal_dim,
                        "total_samples": dataset_size,
                        "train_samples": split_idx,
                        "test_samples": len(signals) - split_idx,
                        "signal_types": ["gaussian_mixture", "exponential_decay", "oscillatory", "sparse"]
                    }
                }
        
        logger.info(f"Generated {len(datasets)} synthetic datasets")
        return datasets
    
    def run_comparative_validation(self, datasets: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Run comprehensive comparative validation across all datasets
        
        Args:
            datasets: Dictionary of datasets to validate on
            
        Returns:
            Comprehensive validation results
        """
        logger.info("Starting comparative validation across all datasets")
        
        comparative_results = {}
        
        for dataset_name, dataset in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            
            training_signals = dataset["training"]
            test_signals = dataset["test"]
            
            # Run comparative analysis
            comparison_result = self.comparative_analyzer.analyze_dataset(
                test_signals=test_signals,
                training_signals=training_signals,
                dataset_name=dataset_name,
                include_adaptation_analysis=True
            )
            
            comparative_results[dataset_name] = comparison_result
            
            # Save intermediate results
            self._save_comparison_results(dataset_name, comparison_result)
        
        return comparative_results
    
    def run_statistical_validation(self, comparative_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run rigorous statistical validation on comparative results
        
        Args:
            comparative_results: Results from comparative analysis
            
        Returns:
            Statistical validation results
        """
        logger.info("Starting statistical validation")
        
        statistical_results = {}
        
        # Aggregate results across datasets for meta-analysis
        all_bioneural_results = []
        all_baseline_results = {}
        
        for dataset_name, comparison_result in comparative_results.items():
            all_bioneural_results.extend(comparison_result.bioneural_results)
            
            for model_name, baseline_results in comparison_result.baseline_results.items():
                if model_name not in all_baseline_results:
                    all_baseline_results[model_name] = []
                all_baseline_results[model_name].extend(baseline_results)
        
        # Convert to format expected by statistical validator
        experimental_data = [
            {
                "processing_time": result.processing_time,
                "overall_quality": result.quality_metrics["overall_quality"],
                "feature_richness": result.quality_metrics["feature_richness"],
                "signal_preservation": result.quality_metrics["signal_preservation"],
                "pattern_complexity": result.bioneural_result.pattern_complexity,
                "fusion_confidence": result.neural_fusion_result.fusion_confidence
            }
            for result in all_bioneural_results
        ]
        
        # Control data from best baseline model (lowest average processing time)
        if all_baseline_results:
            best_baseline = min(all_baseline_results.keys(), 
                              key=lambda k: np.mean([r.processing_time for r in all_baseline_results[k] if np.isfinite(r.processing_time)]))
            
            control_data = [
                {
                    "processing_time": result.processing_time,
                    "model_performance": np.mean(list(result.performance_metrics.values()))
                }
                for result in all_baseline_results[best_baseline]
                if np.isfinite(result.processing_time)
            ]
        else:
            control_data = None
        
        # Run statistical validation
        validation_result = self.statistical_validator.validate_research_claims(
            experimental_data=experimental_data,
            control_data=control_data,
            research_hypotheses=[
                "Bioneural fusion provides significantly better quality than baseline methods",
                "Bioneural fusion maintains competitive processing speed",
                "Bioneural fusion shows improvement through adaptation",
                "Bioneural fusion provides more feature-rich representations"
            ],
            validation_name="bioneural_olfactory_fusion_validation"
        )
        
        statistical_results["meta_analysis"] = validation_result
        
        # Individual dataset validations
        for dataset_name, comparison_result in comparative_results.items():
            dataset_experimental = [
                {
                    "processing_time": result.processing_time,
                    "overall_quality": result.quality_metrics["overall_quality"],
                    "feature_richness": result.quality_metrics["feature_richness"]
                }
                for result in comparison_result.bioneural_results
            ]
            
            dataset_validation = self.statistical_validator.validate_research_claims(
                experimental_data=dataset_experimental,
                validation_name=f"dataset_validation_{dataset_name}"
            )
            
            statistical_results[dataset_name] = dataset_validation
        
        return statistical_results
    
    def run_reproducibility_validation(self, datasets: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Run reproducibility validation across multiple independent runs
        
        Args:
            datasets: Test datasets
            
        Returns:
            Reproducibility validation results
        """
        logger.info(f"Starting reproducibility validation with {self.config['num_validation_runs']} independent runs")
        
        reproducibility_results = {}
        
        # Select a representative dataset for reproducibility testing
        representative_dataset = list(datasets.keys())[0]
        test_signals = datasets[representative_dataset]["test"][:20]  # Use subset for efficiency
        
        # Run multiple independent validations
        run_results = []
        
        for run_idx in range(self.config["num_validation_runs"]):
            logger.info(f"Reproducibility run {run_idx + 1}/{self.config['num_validation_runs']}")
            
            # Create fresh pipeline instance for each run
            pipeline = BioneuralOlfactoryPipeline()
            
            # Process signals
            run_metrics = {
                "processing_times": [],
                "quality_scores": [],
                "pattern_complexities": [],
                "fusion_confidences": []
            }
            
            for signal in test_signals:
                result = pipeline.process(signal)
                run_metrics["processing_times"].append(result.processing_time)
                run_metrics["quality_scores"].append(result.quality_metrics["overall_quality"])
                run_metrics["pattern_complexities"].append(result.bioneural_result.pattern_complexity)
                run_metrics["fusion_confidences"].append(result.neural_fusion_result.fusion_confidence)
            
            # Aggregate run statistics
            run_summary = {
                "run_id": run_idx,
                "avg_processing_time": np.mean(run_metrics["processing_times"]),
                "avg_quality_score": np.mean(run_metrics["quality_scores"]),
                "avg_pattern_complexity": np.mean(run_metrics["pattern_complexities"]),
                "avg_fusion_confidence": np.mean(run_metrics["fusion_confidences"]),
                "std_processing_time": np.std(run_metrics["processing_times"]),
                "std_quality_score": np.std(run_metrics["quality_scores"])
            }
            
            run_results.append(run_summary)
        
        # Analyze reproducibility across runs
        metrics_across_runs = {
            "processing_times": [run["avg_processing_time"] for run in run_results],
            "quality_scores": [run["avg_quality_score"] for run in run_results],
            "pattern_complexities": [run["avg_pattern_complexity"] for run in run_results],
            "fusion_confidences": [run["avg_fusion_confidence"] for run in run_results]
        }
        
        reproducibility_analysis = {}
        for metric_name, values in metrics_across_runs.items():
            reproducibility_analysis[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "coefficient_of_variation": np.std(values) / (np.mean(values) + 1e-10),
                "min": np.min(values),
                "max": np.max(values),
                "range": np.max(values) - np.min(values)
            }
        
        # Overall reproducibility score
        cv_scores = [stats["coefficient_of_variation"] for stats in reproducibility_analysis.values()]
        overall_reproducibility = 1.0 - np.mean(cv_scores)  # Lower CV = higher reproducibility
        
        reproducibility_results = {
            "num_runs": self.config["num_validation_runs"],
            "run_results": run_results,
            "reproducibility_analysis": reproducibility_analysis,
            "overall_reproducibility_score": float(overall_reproducibility),
            "is_reproducible": overall_reproducibility > 0.8,  # 80% reproducibility threshold
            "test_dataset": representative_dataset,
            "num_test_signals": len(test_signals)
        }
        
        return reproducibility_results
    
    def generate_publication_figures(self, 
                                   comparative_results: Dict[str, Any],
                                   statistical_results: Dict[str, Any],
                                   reproducibility_results: Dict[str, Any]) -> None:
        """
        Generate publication-ready figures and visualizations
        
        Args:
            comparative_results: Comparative analysis results
            statistical_results: Statistical validation results
            reproducibility_results: Reproducibility validation results
        """
        logger.info("Generating publication-ready figures")
        
        # Set publication style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Figure 1: Performance Comparison Across Methods
        self._plot_performance_comparison(comparative_results)
        
        # Figure 2: Statistical Significance Results
        self._plot_statistical_significance(statistical_results)
        
        # Figure 3: Effect Sizes and Confidence Intervals
        self._plot_effect_sizes(statistical_results)
        
        # Figure 4: Reproducibility Analysis
        self._plot_reproducibility_analysis(reproducibility_results)
        
        # Figure 5: Scalability Analysis
        self._plot_scalability_analysis(comparative_results)
        
        # Figure 6: Quality Metrics Radar Chart
        self._plot_quality_radar_chart(comparative_results)
        
        logger.info(f"Publication figures saved to {self.output_dir}")
    
    def _plot_performance_comparison(self, comparative_results: Dict[str, Any]) -> None:
        """Plot performance comparison across methods"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Comparison: Bioneural Fusion vs Baseline Methods', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        datasets = list(comparative_results.keys())
        
        # Processing time comparison
        ax1 = axes[0, 0]
        bioneural_times = []
        baseline_times = {}
        
        for dataset_name, result in comparative_results.items():
            bioneural_avg_time = np.mean([r.processing_time for r in result.bioneural_results])
            bioneural_times.append(bioneural_avg_time)
            
            for model_name, baseline_results in result.baseline_results.items():
                if model_name not in baseline_times:
                    baseline_times[model_name] = []
                avg_time = np.mean([r.processing_time for r in baseline_results if np.isfinite(r.processing_time)])
                baseline_times[model_name].append(avg_time)
        
        # Plot processing times
        x = np.arange(len(datasets))
        width = 0.1
        
        ax1.bar(x - width*2, bioneural_times, width, label='Bioneural Fusion', color='red', alpha=0.8)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(baseline_times)))
        for i, (model_name, times) in enumerate(baseline_times.items()):
            ax1.bar(x - width + i*width, times, width, label=model_name, color=colors[i], alpha=0.7)
        
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_xlabel('Dataset')
        ax1.set_title('Processing Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.split('_')[1] for d in datasets], rotation=45)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
        
        # Quality score comparison
        ax2 = axes[0, 1]
        bioneural_qualities = []
        
        for dataset_name, result in comparative_results.items():
            bioneural_avg_quality = np.mean([r.quality_metrics['overall_quality'] for r in result.bioneural_results])
            bioneural_qualities.append(bioneural_avg_quality)
        
        ax2.bar(x, bioneural_qualities, color='green', alpha=0.8, label='Bioneural Fusion')
        ax2.set_ylabel('Average Quality Score')
        ax2.set_xlabel('Dataset')
        ax2.set_title('Quality Score Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.split('_')[1] for d in datasets], rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Adaptation benefit over time
        ax3 = axes[1, 0]
        
        # Use first dataset for adaptation visualization
        first_dataset_result = list(comparative_results.values())[0]
        qualities = [r.quality_metrics['overall_quality'] for r in first_dataset_result.bioneural_results]
        
        ax3.plot(range(len(qualities)), qualities, 'o-', linewidth=2, markersize=6, color='blue')
        ax3.set_xlabel('Processing Sequence')
        ax3.set_ylabel('Quality Score')
        ax3.set_title('Adaptation: Quality Improvement Over Time')
        ax3.grid(alpha=0.3)
        
        # Add trend line
        x_trend = np.arange(len(qualities))
        z = np.polyfit(x_trend, qualities, 1)
        p = np.poly1d(z)
        ax3.plot(x_trend, p(x_trend), "--", alpha=0.8, color='red', label=f'Trend (slope={z[0]:.4f})')
        ax3.legend()
        
        # Feature richness comparison
        ax4 = axes[1, 1]
        bioneural_richness = []
        
        for dataset_name, result in comparative_results.items():
            bioneural_avg_richness = np.mean([r.quality_metrics['feature_richness'] for r in result.bioneural_results])
            bioneural_richness.append(bioneural_avg_richness)
        
        ax4.bar(x, bioneural_richness, color='purple', alpha=0.8)
        ax4.set_ylabel('Feature Richness Score')
        ax4.set_xlabel('Dataset')
        ax4.set_title('Feature Richness Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels([d.split('_')[1] for d in datasets], rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure1_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure1_performance_comparison.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_significance(self, statistical_results: Dict[str, Any]) -> None:
        """Plot statistical significance results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Significance Analysis', fontsize=16, fontweight='bold')
        
        meta_analysis = statistical_results["meta_analysis"]
        
        # P-values heatmap
        ax1 = axes[0, 0]
        
        hypothesis_tests = meta_analysis.hypothesis_tests
        test_names = [test.test_name for test in hypothesis_tests]
        p_values = [test.p_value for test in hypothesis_tests]
        
        # Create p-value categories
        p_categories = []
        colors = []
        for p in p_values:
            if p < 0.001:
                p_categories.append('p < 0.001')
                colors.append('darkgreen')
            elif p < 0.01:
                p_categories.append('p < 0.01')
                colors.append('green')
            elif p < 0.05:
                p_categories.append('p < 0.05')
                colors.append('orange')
            else:
                p_categories.append('p ≥ 0.05')
                colors.append('red')
        
        y_pos = np.arange(len(test_names))
        bars = ax1.barh(y_pos, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([name.replace('_', ' ').title() for name in test_names])
        ax1.set_xlabel('-log₁₀(p-value)')
        ax1.set_title('Statistical Significance Levels')
        ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='α = 0.05')
        ax1.axvline(x=-np.log10(0.01), color='orange', linestyle='--', alpha=0.5, label='α = 0.01')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Effect sizes
        ax2 = axes[0, 1]
        effect_sizes = [test.effect_size for test in hypothesis_tests]
        
        bars = ax2.bar(range(len(effect_sizes)), effect_sizes, 
                      color=['green' if abs(es) > 0.8 else 'orange' if abs(es) > 0.5 else 'gray' 
                            for es in effect_sizes], alpha=0.7)
        
        ax2.set_xlabel('Test Index')
        ax2.set_ylabel('Effect Size (Cohen\\'s d)')
        ax2.set_title('Effect Sizes')
        ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small Effect')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Effect')
        ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Large Effect')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Multiple comparison corrections
        ax3 = axes[1, 0]
        corrections = meta_analysis.multiple_comparison_corrections
        
        correction_labels = ['Uncorrected', 'Bonferroni', 'Benjamini-Hochberg']
        significant_counts = [
            corrections['summary']['uncorrected_significant'],
            corrections['summary']['bonferroni_significant'],
            corrections['summary']['bh_significant']
        ]
        
        bars = ax3.bar(correction_labels, significant_counts, 
                      color=['lightcoral', 'lightblue', 'lightgreen'], alpha=0.8)
        
        ax3.set_ylabel('Number of Significant Tests')
        ax3.set_title('Multiple Comparison Corrections')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, significant_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Power analysis
        ax4 = axes[1, 1]
        power_analyses = meta_analysis.power_analyses
        
        if power_analyses:
            observed_powers = [analysis.power for analysis in power_analyses]
            required_ns = [analysis.required_sample_size for analysis in power_analyses]
            
            ax4_twin = ax4.twinx()
            
            x_power = range(len(observed_powers))
            bars1 = ax4.bar([x - 0.2 for x in x_power], observed_powers, 0.4, 
                           color='blue', alpha=0.7, label='Observed Power')
            ax4_twin.bar([x + 0.2 for x in x_power], required_ns, 0.4, 
                        color='red', alpha=0.7, label='Required N')
            
            ax4.set_ylabel('Statistical Power', color='blue')
            ax4_twin.set_ylabel('Required Sample Size', color='red')
            ax4.set_xlabel('Test Index')
            ax4.set_title('Power Analysis')
            ax4.axhline(y=0.8, color='blue', linestyle='--', alpha=0.5)
            ax4.grid(axis='y', alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax4.text(0.5, 0.5, 'No Power Analysis Data', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure2_statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure2_statistical_significance.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_effect_sizes(self, statistical_results: Dict[str, Any]) -> None:
        """Plot effect sizes with confidence intervals"""
        
        meta_analysis = statistical_results["meta_analysis"]
        hypothesis_tests = meta_analysis.hypothesis_tests
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('Effect Sizes with 95% Confidence Intervals', fontsize=16, fontweight='bold')
        
        test_names = [test.test_name.replace('_', ' ').title() for test in hypothesis_tests]
        effect_sizes = [test.effect_size for test in hypothesis_tests]
        ci_lower = [test.confidence_interval[0] for test in hypothesis_tests if test.confidence_interval[0] is not None]
        ci_upper = [test.confidence_interval[1] for test in hypothesis_tests if test.confidence_interval[1] is not None]
        
        y_pos = np.arange(len(test_names))
        
        # Plot effect sizes as points
        colors = ['green' if abs(es) > 0.8 else 'orange' if abs(es) > 0.5 else 'red' for es in effect_sizes]
        scatter = ax.scatter(effect_sizes, y_pos, c=colors, s=100, alpha=0.8, zorder=3)
        
        # Plot confidence intervals
        if len(ci_lower) == len(effect_sizes) and len(ci_upper) == len(effect_sizes):
            for i, (lower, upper) in enumerate(zip(ci_lower, ci_upper)):
                if lower is not None and upper is not None:
                    ax.plot([lower, upper], [i, i], 'k-', alpha=0.6, linewidth=2)
                    ax.plot([lower, lower], [i-0.1, i+0.1], 'k-', alpha=0.6)
                    ax.plot([upper, upper], [i-0.1, i+0.1], 'k-', alpha=0.6)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(test_names)
        ax.set_xlabel('Effect Size (Cohen\\'s d)')
        ax.set_title('Effect Sizes with 95% Confidence Intervals')
        
        # Add reference lines
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small Effect')
        ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Effect')
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Large Effect')
        ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-0.5, color='orange', linestyle='--', alpha=0.5)
        ax.axvline(x=-0.8, color='green', linestyle='--', alpha=0.5)
        
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure3_effect_sizes.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure3_effect_sizes.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_reproducibility_analysis(self, reproducibility_results: Dict[str, Any]) -> None:
        """Plot reproducibility analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Reproducibility Analysis Across Independent Runs', fontsize=16, fontweight='bold')
        
        run_results = reproducibility_results["run_results"]
        reproducibility_analysis = reproducibility_results["reproducibility_analysis"]
        
        # Processing time consistency
        ax1 = axes[0, 0]
        processing_times = [run["avg_processing_time"] for run in run_results]
        runs = range(1, len(processing_times) + 1)
        
        ax1.plot(runs, processing_times, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.fill_between(runs, 
                        [pt - reproducibility_analysis["processing_times"]["std"] for pt in processing_times],
                        [pt + reproducibility_analysis["processing_times"]["std"] for pt in processing_times],
                        alpha=0.3, color='blue')
        
        ax1.set_xlabel('Independent Run')
        ax1.set_ylabel('Average Processing Time (s)')
        ax1.set_title('Processing Time Consistency')
        ax1.grid(alpha=0.3)
        
        # Add CV annotation
        cv = reproducibility_analysis["processing_times"]["coefficient_of_variation"]
        ax1.text(0.02, 0.98, f'CV = {cv:.3f}', transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Quality score consistency
        ax2 = axes[0, 1]
        quality_scores = [run["avg_quality_score"] for run in run_results]
        
        ax2.plot(runs, quality_scores, 'o-', linewidth=2, markersize=8, color='green')
        ax2.fill_between(runs,
                        [qs - reproducibility_analysis["quality_scores"]["std"] for qs in quality_scores],
                        [qs + reproducibility_analysis["quality_scores"]["std"] for qs in quality_scores],
                        alpha=0.3, color='green')
        
        ax2.set_xlabel('Independent Run')
        ax2.set_ylabel('Average Quality Score')
        ax2.set_title('Quality Score Consistency')
        ax2.grid(alpha=0.3)
        
        # Add CV annotation
        cv_quality = reproducibility_analysis["quality_scores"]["coefficient_of_variation"]
        ax2.text(0.02, 0.98, f'CV = {cv_quality:.3f}', transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Coefficient of variation comparison
        ax3 = axes[1, 0]
        metrics = list(reproducibility_analysis.keys())
        cv_values = [reproducibility_analysis[metric]["coefficient_of_variation"] for metric in metrics]
        
        bars = ax3.bar(range(len(metrics)), cv_values, 
                      color=['blue', 'green', 'orange', 'purple'][:len(metrics)], alpha=0.7)
        
        ax3.set_xticks(range(len(metrics)))
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax3.set_ylabel('Coefficient of Variation')
        ax3.set_title('Reproducibility: Coefficient of Variation')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add threshold line
        ax3.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Good Reproducibility Threshold')
        ax3.legend()
        
        # Overall reproducibility score
        ax4 = axes[1, 1]
        
        overall_score = reproducibility_results["overall_reproducibility_score"]
        is_reproducible = reproducibility_results["is_reproducible"]
        
        # Create a gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax4 = plt.subplot(224, projection='polar')
        ax4.plot(theta, r, 'k-', linewidth=3)
        ax4.fill_between(theta, 0, r, alpha=0.1, color='gray')
        
        # Add score indicator
        score_angle = overall_score * np.pi
        ax4.plot([score_angle, score_angle], [0, 1], 'r-', linewidth=5)
        ax4.plot(score_angle, 1, 'ro', markersize=15)
        
        ax4.set_ylim(0, 1.2)
        ax4.set_theta_zero_location('W')
        ax4.set_theta_direction(1)
        ax4.set_thetagrids([0, 45, 90, 135, 180], ['0', '0.25', '0.5', '0.75', '1.0'])
        ax4.set_rgrids([])
        ax4.set_title(f'Overall Reproducibility Score: {overall_score:.3f}\\n{"✓ Reproducible" if is_reproducible else "✗ Not Reproducible"}', 
                     y=1.1, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure4_reproducibility.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure4_reproducibility.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_scalability_analysis(self, comparative_results: Dict[str, Any]) -> None:
        """Plot scalability analysis across different dataset sizes and dimensions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scalability Analysis: Performance vs Dataset Characteristics', fontsize=16, fontweight='bold')
        
        # Extract scalability data
        dataset_info = {}
        for dataset_name, result in comparative_results.items():
            parts = dataset_name.split('_')
            signal_dim = int(parts[1][:-1])  # Remove 'D'
            dataset_size = int(parts[2][:-7])  # Remove 'samples'
            
            avg_time = np.mean([r.processing_time for r in result.bioneural_results])
            avg_quality = np.mean([r.quality_metrics['overall_quality'] for r in result.bioneural_results])
            
            if signal_dim not in dataset_info:
                dataset_info[signal_dim] = {'sizes': [], 'times': [], 'qualities': []}
            
            dataset_info[signal_dim]['sizes'].append(dataset_size)
            dataset_info[signal_dim]['times'].append(avg_time)
            dataset_info[signal_dim]['qualities'].append(avg_quality)
        
        # Processing time vs signal dimension
        ax1 = axes[0, 0]
        
        dims = sorted(dataset_info.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(dims)))
        
        for dim, color in zip(dims, colors):
            sizes = dataset_info[dim]['sizes']
            times = dataset_info[dim]['times']
            ax1.plot(sizes, times, 'o-', color=color, label=f'{dim}D', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('Processing Time vs Dataset Size')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Quality vs signal dimension
        ax2 = axes[0, 1]
        
        for dim, color in zip(dims, colors):
            sizes = dataset_info[dim]['sizes']
            qualities = dataset_info[dim]['qualities']
            ax2.plot(sizes, qualities, 'o-', color=color, label=f'{dim}D', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Dataset Size')
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Quality Score vs Dataset Size')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_xscale('log')
        
        # Processing time vs signal dimension (fixed dataset size)
        ax3 = axes[1, 0]
        
        # Use largest common dataset size
        common_sizes = set(dataset_info[dims[0]]['sizes'])
        for dim in dims[1:]:
            common_sizes = common_sizes.intersection(set(dataset_info[dim]['sizes']))
        
        if common_sizes:
            target_size = max(common_sizes)
            
            dims_for_scaling = []
            times_for_scaling = []
            
            for dim in dims:
                size_idx = dataset_info[dim]['sizes'].index(target_size)
                dims_for_scaling.append(dim)
                times_for_scaling.append(dataset_info[dim]['times'][size_idx])
            
            ax3.plot(dims_for_scaling, times_for_scaling, 'o-', color='red', linewidth=2, markersize=8)
            ax3.set_xlabel('Signal Dimension')
            ax3.set_ylabel('Processing Time (seconds)')
            ax3.set_title(f'Processing Time vs Signal Dimension\\n(Dataset Size = {target_size})')
            ax3.grid(alpha=0.3)
            
            # Add complexity analysis
            # Fit polynomial to estimate computational complexity
            if len(dims_for_scaling) > 2:
                coeffs = np.polyfit(dims_for_scaling, times_for_scaling, 2)
                dims_smooth = np.linspace(min(dims_for_scaling), max(dims_for_scaling), 100)
                times_fit = np.polyval(coeffs, dims_smooth)
                ax3.plot(dims_smooth, times_fit, '--', color='blue', alpha=0.7, label='Quadratic Fit')
                ax3.legend()
        
        # Memory efficiency analysis (estimated)
        ax4 = axes[1, 1]
        
        # Estimate memory usage based on signal dimensions and processing complexity
        memory_estimates = []
        dims_memory = []
        
        for dim in dims:
            # Simplified memory estimation
            base_memory = dim * 4  # Float32 per signal element
            model_memory = dim * 64 * 4  # Model parameters (rough estimate)
            processing_memory = dim * 128 * 4  # Processing buffers
            
            total_memory_mb = (base_memory + model_memory + processing_memory) / (1024 * 1024)
            
            memory_estimates.append(total_memory_mb)
            dims_memory.append(dim)
        
        ax4.bar(range(len(dims_memory)), memory_estimates, 
               color='orange', alpha=0.7, width=0.6)
        ax4.set_xticks(range(len(dims_memory)))
        ax4.set_xticklabels([f'{d}D' for d in dims_memory])
        ax4.set_ylabel('Estimated Memory Usage (MB)')
        ax4.set_title('Memory Usage vs Signal Dimension')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure5_scalability.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure5_scalability.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_quality_radar_chart(self, comparative_results: Dict[str, Any]) -> None:
        """Plot radar chart comparing quality metrics"""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'))
        fig.suptitle('Quality Metrics Radar Chart: Bioneural Fusion Performance', fontsize=16, fontweight='bold')
        
        # Aggregate quality metrics across all datasets
        all_bioneural_results = []
        for result in comparative_results.values():
            all_bioneural_results.extend(result.bioneural_results)
        
        # Extract quality metrics
        quality_metrics = {
            'Overall Quality': np.mean([r.quality_metrics['overall_quality'] for r in all_bioneural_results]),
            'Feature Richness': np.mean([r.quality_metrics['feature_richness'] for r in all_bioneural_results]),
            'Signal Preservation': np.mean([r.quality_metrics['signal_preservation'] for r in all_bioneural_results]),
            'Processing Consistency': np.mean([r.quality_metrics['processing_consistency'] for r in all_bioneural_results]),
            'Receptor Diversity': np.mean([r.quality_metrics['receptor_diversity'] for r in all_bioneural_results]),
            'Pattern Complexity': np.mean([r.bioneural_result.pattern_complexity for r in all_bioneural_results]),
            'Fusion Confidence': np.mean([r.neural_fusion_result.fusion_confidence for r in all_bioneural_results]),
            'Adaptation Benefit': np.mean([r.quality_metrics.get('adaptation_benefit', 0.5) for r in all_bioneural_results])
        }
        
        # Set up radar chart
        categories = list(quality_metrics.keys())
        values = list(quality_metrics.values())
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values to complete the circle
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=3, color='blue', markersize=8)
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        
        # Set y-axis limits and labels
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True)
        
        # Add value annotations
        for angle, value, category in zip(angles[:-1], values[:-1], categories):
            ax.annotate(f'{value:.3f}', xy=(angle, value), xytext=(10, 10), 
                       textcoords='offset points', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure6_quality_radar.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure6_quality_radar.pdf', bbox_inches='tight')
        plt.close()
    
    def _save_comparison_results(self, dataset_name: str, comparison_result: Any) -> None:
        """Save comparison results to JSON file"""
        
        # Export to JSON-serializable format
        exported_result = self.comparative_analyzer.export_analysis_report(comparison_result)
        
        output_file = self.output_dir / f"comparison_{dataset_name}.json"
        with open(output_file, 'w') as f:
            json.dump(exported_result, f, indent=2, default=str)
    
    def generate_publication_report(self, 
                                  datasets: Dict[str, Dict[str, np.ndarray]],
                                  comparative_results: Dict[str, Any],
                                  statistical_results: Dict[str, Any],
                                  reproducibility_results: Dict[str, Any]) -> None:
        """
        Generate comprehensive publication report
        
        Args:
            datasets: Original datasets used
            comparative_results: Comparative analysis results
            statistical_results: Statistical validation results
            reproducibility_results: Reproducibility validation results
        """
        logger.info("Generating comprehensive publication report")
        
        report = {
            "title": "Bioneural Olfactory Fusion: A Novel Approach for Chemical Signal Processing",
            "abstract": {
                "background": "Traditional olfactory signal processing methods rely on linear transformations that fail to capture the complex, adaptive nature of biological olfactory systems.",
                "methods": "We present a novel bioneural olfactory fusion approach that combines biomimetic receptor ensemble modeling with adaptive neural fusion mechanisms.",
                "results": self._extract_key_results(comparative_results, statistical_results, reproducibility_results),
                "conclusions": self._extract_key_conclusions(statistical_results)
            },
            "experimental_setup": {
                "datasets": {name: data["metadata"] for name, data in datasets.items()},
                "validation_methodology": self.config,
                "statistical_framework": {
                    "alpha": self.config["statistical_alpha"],
                    "effect_size_threshold": self.config["effect_size_threshold"],
                    "power_threshold": self.config["power_threshold"],
                    "multiple_comparison_correction": ["Bonferroni", "Benjamini-Hochberg"],
                    "reproducibility_runs": self.config["num_validation_runs"]
                }
            },
            "results_summary": {
                "comparative_analysis": self._summarize_comparative_results(comparative_results),
                "statistical_validation": self._summarize_statistical_results(statistical_results),
                "reproducibility_assessment": self._summarize_reproducibility_results(reproducibility_results)
            },
            "research_contributions": [
                "Novel biomimetic olfactory receptor ensemble modeling",
                "Adaptive threshold learning with biological inspiration",
                "Multi-scale neural fusion with attention mechanisms",
                "Comprehensive statistical validation framework",
                "Reproducible research methodology and code"
            ],
            "limitations_and_future_work": [
                "Synthetic dataset validation - requires real olfactory data validation",
                "Computational complexity analysis for larger-scale applications",
                "Integration with existing olfactory sensing hardware",
                "Cross-species olfactory model validation"
            ],
            "metadata": {
                "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tests_performed": sum(len(sr.hypothesis_tests) for sr in statistical_results.values() if hasattr(sr, 'hypothesis_tests')),
                "total_datasets_analyzed": len(datasets),
                "total_signals_processed": sum(data["metadata"]["total_samples"] for data in datasets.values())
            }
        }
        
        # Save comprehensive report
        report_file = self.output_dir / "publication_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown summary
        self._generate_markdown_summary(report)
        
        logger.info(f"Publication report saved to {report_file}")
    
    def _extract_key_results(self, comparative_results, statistical_results, reproducibility_results):
        """Extract key results for abstract"""
        
        # Count significant improvements
        meta_analysis = statistical_results.get("meta_analysis")
        if meta_analysis:
            significant_tests = sum(1 for test in meta_analysis.hypothesis_tests if test.statistical_significance)
            large_effects = sum(1 for test in meta_analysis.hypothesis_tests if abs(test.effect_size) > 0.8)
        else:
            significant_tests, large_effects = 0, 0
        
        # Average quality improvement
        all_quality_scores = []
        for result in comparative_results.values():
            all_quality_scores.extend([r.quality_metrics['overall_quality'] for r in result.bioneural_results])
        avg_quality = np.mean(all_quality_scores)
        
        # Reproducibility score
        reproducibility_score = reproducibility_results.get("overall_reproducibility_score", 0.0)
        
        return {
            "significant_improvements": significant_tests,
            "large_effect_sizes": large_effects,
            "average_quality_score": float(avg_quality),
            "reproducibility_score": float(reproducibility_score),
            "datasets_validated": len(comparative_results)
        }
    
    def _extract_key_conclusions(self, statistical_results):
        """Extract key conclusions from statistical validation"""
        
        conclusions = []
        
        meta_analysis = statistical_results.get("meta_analysis")
        if meta_analysis and hasattr(meta_analysis, 'summary_conclusions'):
            conclusions = meta_analysis.summary_conclusions[:3]  # Top 3 conclusions
        
        return conclusions
    
    def _summarize_comparative_results(self, comparative_results):
        """Summarize comparative analysis results"""
        
        summary = {
            "datasets_analyzed": len(comparative_results),
            "avg_processing_time": 0.0,
            "avg_quality_score": 0.0,
            "best_performing_dataset": "",
            "adaptation_observed": False
        }
        
        all_times = []
        all_qualities = []
        dataset_qualities = {}
        
        for dataset_name, result in comparative_results.items():
            times = [r.processing_time for r in result.bioneural_results]
            qualities = [r.quality_metrics['overall_quality'] for r in result.bioneural_results]
            
            all_times.extend(times)
            all_qualities.extend(qualities)
            dataset_qualities[dataset_name] = np.mean(qualities)
            
            # Check for adaptation (quality improvement over sequence)
            if len(qualities) > 1:
                from scipy import stats
                slope, _, _, p_value, _ = stats.linregress(range(len(qualities)), qualities)
                if slope > 0 and p_value < 0.05:
                    summary["adaptation_observed"] = True
        
        summary["avg_processing_time"] = float(np.mean(all_times))
        summary["avg_quality_score"] = float(np.mean(all_qualities))
        summary["best_performing_dataset"] = max(dataset_qualities.keys(), key=lambda k: dataset_qualities[k])
        
        return summary
    
    def _summarize_statistical_results(self, statistical_results):
        """Summarize statistical validation results"""
        
        meta_analysis = statistical_results.get("meta_analysis")
        if not meta_analysis:
            return {"no_meta_analysis": "Meta-analysis not available"}
        
        hypothesis_tests = meta_analysis.hypothesis_tests
        corrections = meta_analysis.multiple_comparison_corrections
        
        summary = {
            "total_hypothesis_tests": len(hypothesis_tests),
            "statistically_significant": sum(1 for test in hypothesis_tests if test.statistical_significance),
            "practically_significant": sum(1 for test in hypothesis_tests if test.practical_significance),
            "large_effect_sizes": sum(1 for test in hypothesis_tests if abs(test.effect_size) > 0.8),
            "bonferroni_significant": corrections["summary"]["bonferroni_significant"],
            "fdr_significant": corrections["summary"]["bh_significant"],
            "avg_effect_size": float(np.mean([test.effect_size for test in hypothesis_tests])),
            "min_p_value": float(min(test.p_value for test in hypothesis_tests)) if hypothesis_tests else 1.0
        }
        
        return summary
    
    def _summarize_reproducibility_results(self, reproducibility_results):
        """Summarize reproducibility validation results"""
        
        return {
            "independent_runs": reproducibility_results["num_runs"],
            "overall_reproducibility_score": reproducibility_results["overall_reproducibility_score"],
            "is_reproducible": reproducibility_results["is_reproducible"],
            "most_consistent_metric": min(reproducibility_results["reproducibility_analysis"].keys(),
                                        key=lambda k: reproducibility_results["reproducibility_analysis"][k]["coefficient_of_variation"]),
            "least_consistent_metric": max(reproducibility_results["reproducibility_analysis"].keys(),
                                         key=lambda k: reproducibility_results["reproducibility_analysis"][k]["coefficient_of_variation"])
        }
    
    def _generate_markdown_summary(self, report):
        """Generate markdown summary report"""
        
        markdown_content = f"""# {report['title']}

## Abstract

**Background:** {report['abstract']['background']}

**Methods:** {report['abstract']['methods']}

**Results:** 
- Significant improvements: {report['abstract']['results']['significant_improvements']} tests
- Large effect sizes: {report['abstract']['results']['large_effect_sizes']} tests  
- Average quality score: {report['abstract']['results']['average_quality_score']:.3f}
- Reproducibility score: {report['abstract']['results']['reproducibility_score']:.3f}
- Datasets validated: {report['abstract']['results']['datasets_validated']}

**Conclusions:** {' '.join(report['abstract']['conclusions'])}

## Experimental Setup

### Datasets
"""
        
        for dataset_name, metadata in report['experimental_setup']['datasets'].items():
            markdown_content += f"- **{dataset_name}**: {metadata['total_samples']} samples, {metadata['signal_dim']}D signals\\n"
        
        markdown_content += f"""
### Statistical Framework
- Significance level (α): {report['experimental_setup']['statistical_framework']['alpha']}
- Effect size threshold: {report['experimental_setup']['statistical_framework']['effect_size_threshold']}
- Power threshold: {report['experimental_setup']['statistical_framework']['power_threshold']}
- Multiple comparison corrections: {', '.join(report['experimental_setup']['statistical_framework']['multiple_comparison_correction'])}
- Reproducibility runs: {report['experimental_setup']['statistical_framework']['reproducibility_runs']}

## Key Results

### Comparative Analysis
- Datasets analyzed: {report['results_summary']['comparative_analysis']['datasets_analyzed']}
- Average processing time: {report['results_summary']['comparative_analysis']['avg_processing_time']:.4f}s
- Average quality score: {report['results_summary']['comparative_analysis']['avg_quality_score']:.3f}
- Best performing dataset: {report['results_summary']['comparative_analysis']['best_performing_dataset']}
- Adaptation observed: {'Yes' if report['results_summary']['comparative_analysis']['adaptation_observed'] else 'No'}

### Statistical Validation
- Total hypothesis tests: {report['results_summary']['statistical_validation']['total_hypothesis_tests']}
- Statistically significant: {report['results_summary']['statistical_validation']['statistically_significant']}
- Practically significant: {report['results_summary']['statistical_validation']['practically_significant']}
- Large effect sizes: {report['results_summary']['statistical_validation']['large_effect_sizes']}
- Bonferroni significant: {report['results_summary']['statistical_validation']['bonferroni_significant']}
- FDR significant: {report['results_summary']['statistical_validation']['fdr_significant']}
- Average effect size: {report['results_summary']['statistical_validation']['avg_effect_size']:.3f}

### Reproducibility Assessment
- Independent runs: {report['results_summary']['reproducibility_assessment']['independent_runs']}
- Reproducibility score: {report['results_summary']['reproducibility_assessment']['overall_reproducibility_score']:.3f}
- Is reproducible: {'Yes' if report['results_summary']['reproducibility_assessment']['is_reproducible'] else 'No'}
- Most consistent metric: {report['results_summary']['reproducibility_assessment']['most_consistent_metric']}

## Research Contributions
"""
        
        for contribution in report['research_contributions']:
            markdown_content += f"- {contribution}\\n"
        
        markdown_content += f"""
## Limitations and Future Work
"""
        
        for limitation in report['limitations_and_future_work']:
            markdown_content += f"- {limitation}\\n"
        
        markdown_content += f"""
## Validation Metadata
- Validation date: {report['metadata']['validation_date']}
- Total tests performed: {report['metadata']['total_tests_performed']}
- Total datasets analyzed: {report['metadata']['total_datasets_analyzed']}
- Total signals processed: {report['metadata']['total_signals_processed']}

---
*This report was generated automatically by the Research Validation Suite.*
"""
        
        # Save markdown report
        markdown_file = self.output_dir / "RESEARCH_SUMMARY.md"
        with open(markdown_file, 'w') as f:
            f.write(markdown_content)


def main():
    """Main function for running research validation suite"""
    
    parser = argparse.ArgumentParser(description='Bioneural Olfactory Fusion Research Validation Suite')
    parser.add_argument('--output-dir', default='research_results', help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--quick', action='store_true', help='Run quick validation with reduced parameters')
    
    args = parser.parse_args()
    
    # Configure validation for quick run
    if args.quick:
        validation_config = {
            "signal_dimensions": [64, 128],
            "dataset_sizes": [30, 50],
            "num_validation_runs": 3
        }
    else:
        validation_config = None
    
    # Initialize validation suite
    suite = ResearchValidationSuite(
        output_dir=args.output_dir,
        random_seed=args.seed,
        validation_config=validation_config
    )
    
    print("🧪 Starting Comprehensive Research Validation Suite")
    print("=" * 60)
    
    # Step 1: Generate synthetic datasets
    print("📊 Step 1: Generating synthetic datasets...")
    datasets = suite.generate_synthetic_datasets()
    print(f"✅ Generated {len(datasets)} datasets")
    
    # Step 2: Run comparative validation
    print("🔬 Step 2: Running comparative validation...")
    comparative_results = suite.run_comparative_validation(datasets)
    print("✅ Comparative analysis complete")
    
    # Step 3: Run statistical validation
    print("📈 Step 3: Running statistical validation...")
    statistical_results = suite.run_statistical_validation(comparative_results)
    print("✅ Statistical validation complete")
    
    # Step 4: Run reproducibility validation
    print("🔄 Step 4: Running reproducibility validation...")
    reproducibility_results = suite.run_reproducibility_validation(datasets)
    print("✅ Reproducibility validation complete")
    
    # Step 5: Generate publication figures
    print("📊 Step 5: Generating publication figures...")
    suite.generate_publication_figures(comparative_results, statistical_results, reproducibility_results)
    print("✅ Publication figures generated")
    
    # Step 6: Generate publication report
    print("📄 Step 6: Generating publication report...")
    suite.generate_publication_report(datasets, comparative_results, statistical_results, reproducibility_results)
    print("✅ Publication report generated")
    
    print("=" * 60)
    print(f"🎉 Research validation complete! Results saved to: {args.output_dir}")
    print(f"📊 View figures: {args.output_dir}/*.png")
    print(f"📄 View report: {args.output_dir}/RESEARCH_SUMMARY.md")
    print("=" * 60)


if __name__ == "__main__":
    main()