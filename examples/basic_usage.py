#!/usr/bin/env python3
"""Basic usage example of the AI Science Platform"""

import numpy as np
from src.algorithms.discovery import DiscoveryEngine
from src.experiments.runner import ExperimentRunner, ExperimentConfig
from src.utils.data_utils import generate_sample_data, validate_data
from src.utils.visualization import plot_discovery_results, plot_data_distribution


def main():
    """Demonstrate basic platform usage"""
    print("🧬 AI Science Platform - Basic Usage Example")
    print("=" * 50)
    
    # 1. Generate sample data
    print("\n1. Generating sample data...")
    data, targets = generate_sample_data(
        size=1000, 
        data_type="sine",
        frequency=2.0,
        amplitude=3.0,
        noise=0.2,
        seed=42
    )
    print(f"Generated data shape: {data.shape}, targets shape: {targets.shape}")
    
    # 2. Validate data quality
    print("\n2. Validating data quality...")
    validation = validate_data(data, targets)
    print(f"Data valid: {validation['valid']}")
    if validation['issues']:
        print(f"Issues found: {validation['issues']}")
    print(f"Data statistics: {validation['statistics']['n_samples']} samples, "
          f"{validation['statistics']['n_features']} features")
    
    # 3. Initialize discovery engine
    print("\n3. Initializing discovery engine...")
    engine = DiscoveryEngine(discovery_threshold=0.6)
    
    # 4. Run discovery process
    print("\n4. Running scientific discovery...")
    discoveries = engine.discover(data, targets, context="sine_wave_analysis")
    
    print(f"Found {len(discoveries)} discoveries!")
    for i, discovery in enumerate(discoveries):
        print(f"  Discovery {i+1}:")
        print(f"    Hypothesis: {discovery.hypothesis}")
        print(f"    Confidence: {discovery.confidence:.3f}")
        print(f"    Key metrics: {discovery.metrics}")
    
    # 5. Get engine summary
    print("\n5. Discovery engine summary:")
    summary = engine.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 6. Setup experiment runner
    print("\n6. Setting up systematic experiments...")
    experiment_runner = ExperimentRunner("./experiment_results")
    
    # Define experiment configuration
    config = ExperimentConfig(
        name="discovery_optimization",
        description="Optimize discovery parameters",
        parameters={
            "threshold": 0.6,
            "data_size": 1000
        },
        metrics_to_track=["num_discoveries", "avg_confidence", "processing_time"],
        num_runs=3,
        seed=42
    )
    
    experiment_runner.register_experiment(config)
    
    # Define experiment function
    def discovery_experiment(params):
        import time
        start_time = time.time()
        
        # Generate data
        exp_data, exp_targets = generate_sample_data(
            size=params["data_size"],
            data_type="polynomial",
            degree=2,
            noise=0.15,
            seed=np.random.randint(0, 1000)
        )
        
        # Run discovery
        exp_engine = DiscoveryEngine(discovery_threshold=params["threshold"])
        exp_discoveries = exp_engine.discover(exp_data, exp_targets)
        
        processing_time = time.time() - start_time
        
        return {
            "num_discoveries": len(exp_discoveries),
            "avg_confidence": np.mean([d.confidence for d in exp_discoveries]) if exp_discoveries else 0,
            "processing_time": processing_time
        }
    
    # 7. Run experiments
    print("\n7. Running systematic experiments...")
    exp_results = experiment_runner.run_experiment(
        "discovery_optimization", 
        discovery_experiment
    )
    
    # 8. Analyze results
    print("\n8. Analyzing experiment results...")
    analysis = experiment_runner.analyze_results("discovery_optimization")
    
    print(f"Experiment: {analysis['experiment_name']}")
    print(f"Success rate: {analysis['success_rate']:.1%}")
    print(f"Average execution time: {analysis['avg_execution_time']:.3f}s")
    
    print("\nMetrics summary:")
    for metric, stats in analysis['metrics_summary'].items():
        print(f"  {metric}:")
        print(f"    Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    # 9. Generate report
    print("\n9. Generating comprehensive report...")
    report = experiment_runner.generate_report("discovery_optimization")
    print("Report preview (first 300 chars):")
    print(report[:300] + "..." if len(report) > 300 else report)
    
    # 10. Demonstrate visualization (without showing)
    print("\n10. Creating visualizations...")
    try:
        # Discovery results plot
        fig1 = plot_discovery_results(discoveries, "discovery_results.png")
        print("Discovery results plot saved to: discovery_results.png")
        
        # Data distribution plot  
        fig2 = plot_data_distribution(data, targets, "data_distribution.png")
        print("Data distribution plot saved to: data_distribution.png")
        
    except ImportError:
        print("Visualization skipped (matplotlib not available in test environment)")
    
    print("\n✅ Basic usage demonstration complete!")
    print("\nNext steps:")
    print("- Explore different data types and discovery parameters")
    print("- Implement custom models using BaseModel")
    print("- Scale up with larger datasets")
    print("- Integrate with your scientific workflow")


if __name__ == "__main__":
    main()