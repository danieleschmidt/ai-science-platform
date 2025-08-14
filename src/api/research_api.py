"""Research API for scientific discovery automation"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging
from flask import Flask, request, jsonify
from ..algorithms.discovery import DiscoveryEngine
from ..experiments.runner import ExperimentRunner, ExperimentConfig
from ..algorithms.bioneural_pipeline import BioneuralOlfactoryPipeline
from ..utils.error_handling import safe_array_operation, robust_execution

logger = logging.getLogger(__name__)


class ResearchAPI:
    """REST API for research operations"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        
        # Initialize core components
        self.discovery_engine = DiscoveryEngine()
        self.experiment_runner = ExperimentRunner()
        self.bioneural_pipeline = BioneuralOlfactoryPipeline()
        
        self._setup_routes()
        logger.info("ResearchAPI initialized")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy", "service": "ai-science-platform"})
        
        @self.app.route('/discover', methods=['POST'])
        @robust_execution(recovery_strategy='graceful_degradation')
        def discover():
            """Generate scientific discoveries from data"""
            try:
                data = request.json
                if 'data' not in data:
                    return jsonify({"error": "Missing 'data' field"}), 400
                
                input_data = np.array(data['data'])
                context = data.get('context', '')
                targets = np.array(data['targets']) if 'targets' in data else None
                
                discoveries = self.discovery_engine.discover(input_data, targets, context)
                
                return jsonify({
                    "discoveries": [
                        {
                            "hypothesis": d.hypothesis,
                            "confidence": d.confidence,
                            "metrics": d.metrics,
                            "timestamp": d.timestamp
                        } for d in discoveries
                    ],
                    "summary": self.discovery_engine.summary()
                })
                
            except Exception as e:
                logger.error(f"Discovery error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/experiment', methods=['POST'])
        @robust_execution(recovery_strategy='graceful_degradation')
        def run_experiment():
            """Run a scientific experiment"""
            try:
                data = request.json
                exp_name = data.get('name', 'default_experiment')
                description = data.get('description', 'API experiment')
                parameters = data.get('parameters', {})
                num_runs = data.get('num_runs', 3)
                
                # Register experiment
                config = ExperimentConfig(
                    name=exp_name,
                    description=description,
                    parameters=parameters,
                    metrics_to_track=['accuracy', 'confidence'],
                    num_runs=num_runs
                )
                self.experiment_runner.register_experiment(config)
                
                # Define simple experiment function
                def simple_experiment(params):
                    input_data = np.array(params.get('data', [1, 2, 3, 4, 5]))
                    discoveries = self.discovery_engine.discover(input_data)
                    
                    return {
                        'accuracy': min(1.0, len(discoveries) * 0.3 + 0.5),
                        'confidence': np.mean([d.confidence for d in discoveries]) if discoveries else 0.5
                    }
                
                # Run experiment
                results = self.experiment_runner.run_experiment(exp_name, simple_experiment)
                analysis = self.experiment_runner.analyze_results(exp_name)
                
                return jsonify({
                    "experiment_name": exp_name,
                    "results": [
                        {
                            "run_id": r.run_id,
                            "metrics": r.metrics,
                            "success": r.success,
                            "execution_time": r.execution_time
                        } for r in results
                    ],
                    "analysis": analysis
                })
                
            except Exception as e:
                logger.error(f"Experiment error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/bioneural', methods=['POST'])
        @robust_execution(recovery_strategy='graceful_degradation')
        def bioneural_process():
            """Process signal through bioneural pipeline"""
            try:
                data = request.json
                if 'signal' not in data:
                    return jsonify({"error": "Missing 'signal' field"}), 400
                
                signal = np.array(data['signal'])
                metadata = data.get('metadata', {})
                
                result = self.bioneural_pipeline.process(signal, metadata)
                exported_result = self.bioneural_pipeline.export_processing_result(result)
                
                return jsonify({
                    "processing_result": exported_result,
                    "pipeline_summary": self.bioneural_pipeline.summary()
                })
                
            except Exception as e:
                logger.error(f"Bioneural processing error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """Get system status"""
            return jsonify({
                "discovery_engine": self.discovery_engine.summary(),
                "bioneural_pipeline": self.bioneural_pipeline.get_component_summaries(),
                "api_info": {
                    "host": self.host,
                    "port": self.port,
                    "endpoints": ["/health", "/discover", "/experiment", "/bioneural", "/status"]
                }
            })
    
    def run(self, debug: bool = False):
        """Run the API server"""
        logger.info(f"Starting ResearchAPI on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)
    
    def get_app(self):
        """Get Flask app for testing"""
        return self.app