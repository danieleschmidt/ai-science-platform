"""Discovery-focused API endpoints"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from flask import Flask, request, jsonify, Blueprint
from ..algorithms.discovery import DiscoveryEngine
from ..models.simple import SimpleModel, SimpleDiscoveryModel
from ..utils.error_handling import robust_execution
from ..utils.validation import ValidationMixin

logger = logging.getLogger(__name__)


class DiscoveryAPI(ValidationMixin):
    """Specialized API for discovery operations"""
    
    def __init__(self):
        self.blueprint = Blueprint('discovery', __name__)
        self.discovery_engine = DiscoveryEngine()
        self.model = SimpleDiscoveryModel()
        self._setup_routes()
        logger.info("DiscoveryAPI initialized")
    
    def _setup_routes(self):
        """Setup discovery-specific routes"""
        
        @self.blueprint.route('/generate_hypothesis', methods=['POST'])
        @robust_execution(recovery_strategy='graceful_degradation')
        def generate_hypothesis():
            """Generate scientific hypothesis from data"""
            try:
                data = request.json
                if 'data' not in data:
                    return jsonify({"error": "Missing 'data' field"}), 400
                
                input_data = np.array(data['data'])
                context = data.get('context', '')
                
                hypothesis = self.discovery_engine.generate_hypothesis(input_data, context)
                
                return jsonify({
                    "hypothesis": hypothesis,
                    "data_summary": {
                        "shape": input_data.shape,
                        "mean": float(np.mean(input_data)),
                        "std": float(np.std(input_data))
                    },
                    "context": context
                })
                
            except Exception as e:
                logger.error(f"Hypothesis generation error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.blueprint.route('/test_hypothesis', methods=['POST'])
        @robust_execution(recovery_strategy='graceful_degradation')
        def test_hypothesis():
            """Test a scientific hypothesis"""
            try:
                data = request.json
                if 'hypothesis' not in data or 'data' not in data:
                    return jsonify({"error": "Missing 'hypothesis' or 'data' field"}), 400
                
                hypothesis = data['hypothesis']
                input_data = np.array(data['data'])
                targets = np.array(data['targets']) if 'targets' in data else None
                
                is_valid, metrics = self.discovery_engine.test_hypothesis(
                    hypothesis, input_data, targets
                )
                
                return jsonify({
                    "hypothesis": hypothesis,
                    "is_valid": is_valid,
                    "metrics": metrics,
                    "validation_summary": {
                        "data_size": len(input_data),
                        "targets_provided": targets is not None
                    }
                })
                
            except Exception as e:
                logger.error(f"Hypothesis testing error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.blueprint.route('/discover_patterns', methods=['POST'])
        @robust_execution(recovery_strategy='graceful_degradation')
        def discover_patterns():
            """Discover patterns in data using AI models"""
            try:
                data = request.json
                if 'data' not in data:
                    return jsonify({"error": "Missing 'data' field"}), 400
                
                input_data = np.array(data['data'])
                
                # Train discovery model
                self.model.fit(input_data)
                
                # Extract patterns
                patterns = self.model.get_patterns()
                
                # Make predictions
                predictions = self.model.predict(input_data)
                
                return jsonify({
                    "patterns_discovered": len(patterns),
                    "pattern_summaries": [
                        {
                            "pattern_id": i,
                            "shape": pattern.shape if hasattr(pattern, 'shape') else len(pattern),
                            "mean": float(np.mean(pattern)),
                            "std": float(np.std(pattern))
                        } for i, pattern in enumerate(patterns)
                    ],
                    "predictions": predictions.tolist(),
                    "model_summary": self.model.summary()
                })
                
            except Exception as e:
                logger.error(f"Pattern discovery error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.blueprint.route('/best_discoveries', methods=['GET'])
        def get_best_discoveries():
            """Get top discoveries by confidence"""
            try:
                top_k = request.args.get('top_k', 5, type=int)
                discoveries = self.discovery_engine.get_best_discoveries(top_k)
                
                return jsonify({
                    "top_discoveries": [
                        {
                            "hypothesis": d.hypothesis,
                            "confidence": d.confidence,
                            "metrics": d.metrics,
                            "timestamp": d.timestamp
                        } for d in discoveries
                    ],
                    "total_discoveries": len(self.discovery_engine.discoveries),
                    "engine_summary": self.discovery_engine.summary()
                })
                
            except Exception as e:
                logger.error(f"Best discoveries error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.blueprint.route('/batch_discover', methods=['POST'])
        @robust_execution(recovery_strategy='graceful_degradation')
        def batch_discover():
            """Process multiple datasets for discovery"""
            try:
                data = request.json
                if 'datasets' not in data:
                    return jsonify({"error": "Missing 'datasets' field"}), 400
                
                datasets = data['datasets']
                results = []
                
                for i, dataset in enumerate(datasets):
                    input_data = np.array(dataset['data'])
                    context = dataset.get('context', f'dataset_{i}')
                    targets = np.array(dataset['targets']) if 'targets' in dataset else None
                    
                    discoveries = self.discovery_engine.discover(input_data, targets, context)
                    
                    results.append({
                        "dataset_id": i,
                        "context": context,
                        "discoveries_count": len(discoveries),
                        "best_confidence": max([d.confidence for d in discoveries]) if discoveries else 0.0,
                        "discoveries": [
                            {
                                "hypothesis": d.hypothesis,
                                "confidence": d.confidence,
                                "metrics": d.metrics
                            } for d in discoveries
                        ]
                    })
                
                return jsonify({
                    "batch_results": results,
                    "total_datasets": len(datasets),
                    "summary": self.discovery_engine.summary()
                })
                
            except Exception as e:
                logger.error(f"Batch discovery error: {e}")
                return jsonify({"error": str(e)}), 500
    
    def get_blueprint(self) -> Blueprint:
        """Get Flask blueprint"""
        return self.blueprint