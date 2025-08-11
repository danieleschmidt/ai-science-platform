"""Command Line Interface for AI Science Platform"""

import argparse
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from .algorithms.discovery import DiscoveryEngine
from . import DiscoveryModel
from .experiments.runner import ExperimentRunner, ExperimentConfig
from .utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="AI Science Platform CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Discovery command
    discovery_parser = subparsers.add_parser('discover', help='Run discovery engine')
    discovery_parser.add_argument('--data', type=str, help='Path to data file')
    discovery_parser.add_argument('--threshold', type=float, default=0.7, help='Discovery threshold')
    discovery_parser.add_argument('--output', type=str, help='Output file for results')
    
    # Model command
    model_parser = subparsers.add_parser('model', help='Train and use models')
    model_parser.add_argument('--action', choices=['train', 'predict'], required=True)
    model_parser.add_argument('--data', type=str, help='Path to data file')
    model_parser.add_argument('--model-type', choices=['discovery', 'scientific'], default='discovery')
    model_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    
    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run experiments')
    exp_parser.add_argument('--name', type=str, required=True, help='Experiment name')
    exp_parser.add_argument('--data', type=str, help='Path to data file')
    exp_parser.add_argument('--runs', type=int, default=3, help='Number of runs')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show platform status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging()
    
    try:
        if args.command == 'discover':
            return run_discovery(args)
        elif args.command == 'model':
            return run_model(args)
        elif args.command == 'experiment':
            return run_experiment(args)
        elif args.command == 'status':
            return show_status(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1


def run_discovery(args) -> int:
    """Run discovery engine"""
    if not args.data:
        # Generate sample data for demonstration
        data = np.random.randn(100, 5)
        print("No data file provided, using generated sample data")
    else:
        try:
            data = np.loadtxt(args.data)
        except Exception as e:
            print(f"Error loading data: {e}")
            return 1
    
    engine = DiscoveryEngine(discovery_threshold=args.threshold)
    discoveries = engine.discover(data, context="CLI_discovery")
    
    print(f"\n🔬 Discovery Results:")
    print(f"Discoveries found: {len(discoveries)}")
    
    for i, discovery in enumerate(discoveries):
        print(f"\nDiscovery {i+1}:")
        print(f"  Hypothesis: {discovery.hypothesis}")
        print(f"  Confidence: {discovery.confidence:.3f}")
        print(f"  Evidence: {len(discovery.evidence)} data points")
    
    if args.output:
        # Save results (simplified format)
        with open(args.output, 'w') as f:
            f.write(f"Discovery Results ({len(discoveries)} found)\n")
            for i, discovery in enumerate(discoveries):
                f.write(f"\nDiscovery {i+1}:\n")
                f.write(f"Hypothesis: {discovery.hypothesis}\n")
                f.write(f"Confidence: {discovery.confidence:.3f}\n")
        print(f"Results saved to {args.output}")
    
    return 0


def run_model(args) -> int:
    """Train or use models"""
    if not args.data:
        # Generate sample data
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 1)
        print("No data file provided, using generated sample data")
    else:
        try:
            data = np.loadtxt(args.data)
            X = data[:, :-1]
            y = data[:, -1:]
        except Exception as e:
            print(f"Error loading data: {e}")
            return 1
    
    if args.model_type == 'discovery':
        model = DiscoveryModel(input_dim=X.shape[1])
    else:
        # Use simple model for scientific tasks too
        model = DiscoveryModel(input_dim=X.shape[1])
    
    if args.action == 'train':
        print(f"Training {args.model_type} model...")
        history = model.fit(X, y, epochs=args.epochs)
        print(f"Training completed. Final loss: {history['loss'][-1]:.6f}")
        
        # Save model
        model_path = f"{args.model_type}_model.pth"
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
        
    elif args.action == 'predict':
        predictions = model.predict(X[:10])  # Predict on first 10 samples
        print(f"Predictions for first 10 samples:")
        for i, pred in enumerate(predictions):
            print(f"  Sample {i+1}: {pred}")
    
    return 0


def run_experiment(args) -> int:
    """Run structured experiments"""
    if not args.data:
        data = np.random.randn(200, 8)
        print("No data file provided, using generated sample data")
    else:
        try:
            data = np.loadtxt(args.data)
        except Exception as e:
            print(f"Error loading data: {e}")
            return 1
    
    runner = ExperimentRunner()
    
    # Define a simple experiment
    config = ExperimentConfig(
        name=args.name,
        description=f"CLI experiment: {args.name}",
        parameters={"threshold": 0.7, "method": "discovery"},
        metrics_to_track=["accuracy", "confidence", "discoveries"],
        num_runs=args.runs,
        seed=42
    )
    
    runner.register_experiment(config)
    
    def experiment_function(params):
        """Simple experiment function"""
        engine = DiscoveryEngine(discovery_threshold=params["threshold"])
        discoveries = engine.discover(params["data"], context=args.name)
        
        return {
            "discoveries": len(discoveries),
            "confidence": np.mean([d.confidence for d in discoveries]) if discoveries else 0.0,
            "hypotheses_tested": engine.hypotheses_tested
        }
    
    print(f"Running experiment: {args.name}")
    results = runner.run_experiment(args.name, experiment_function, data)
    
    # Show analysis
    analysis = runner.analyze_results(args.name)
    print(f"\n📊 Experiment Analysis:")
    print(f"Success rate: {analysis['success_rate']:.2%}")
    print(f"Average execution time: {analysis['avg_execution_time']:.3f}s")
    
    for metric, summary in analysis['metrics_summary'].items():
        print(f"{metric}: {summary['mean']:.3f} ± {summary['std']:.3f}")
    
    return 0


def show_status(args) -> int:
    """Show platform status"""
    print("🔬 AI Science Platform Status")
    print("=" * 40)
    
    try:
        # Test components
        engine = DiscoveryEngine()
        model = DiscoveryModel(input_dim=5)
        runner = ExperimentRunner()
        
        print("✅ Discovery Engine: Ready")
        print("✅ Discovery Model: Ready") 
        print("✅ Experiment Runner: Ready")
        
        # Show capabilities
        print(f"\nDiscovery Engine Summary:")
        summary = engine.summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
            
        print(f"\nDiscovery Model Summary:")
        model_info = model.get_discovery_summary()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return 1
    
    print("\n🚀 Platform is operational!")
    return 0


if __name__ == "__main__":
    sys.exit(main())