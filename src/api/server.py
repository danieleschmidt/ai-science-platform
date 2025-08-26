"""AI Science Platform API Server"""

import logging
import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path

# Simple HTTP server implementation (no Flask dependency)
import http.server
import socketserver
import json
import urllib.parse
from datetime import datetime

from ..algorithms.discovery import DiscoveryEngine
from ..models.simple import SimpleDiscoveryModel
from ..experiments.runner import ExperimentRunner
from ..utils.logging_config import setup_logging
from ..utils.data_utils import generate_sample_data

logger = logging.getLogger(__name__)


class AIScientificPlatformHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for AI Science Platform API"""
    
    def __init__(self, *args, **kwargs):
        self.discovery_engine = DiscoveryEngine()
        self.experiment_runner = ExperimentRunner()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path
            
            if path == '/':
                self._handle_root()
            elif path == '/health':
                self._handle_health()
            elif path == '/status':
                self._handle_status()
            elif path == '/api/discover':
                self._handle_discovery()
            elif path == '/api/experiments':
                self._handle_list_experiments()
            else:
                self._send_error(404, "Endpoint not found")
                
        except Exception as e:
            logger.error(f"GET request error: {e}")
            self._send_error(500, str(e))
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8')) if post_data else {}
            except json.JSONDecodeError:
                self._send_error(400, "Invalid JSON in request body")
                return
            
            if path == '/api/discover':
                self._handle_discovery_post(data)
            elif path == '/api/model/predict':
                self._handle_model_prediction(data)
            elif path == '/api/experiments/run':
                self._handle_run_experiment(data)
            else:
                self._send_error(404, "Endpoint not found")
                
        except Exception as e:
            logger.error(f"POST request error: {e}")
            self._send_error(500, str(e))
    
    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response"""
        response = json.dumps(data, indent=2, default=str)
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def _send_error(self, status_code: int, message: str):
        """Send error response"""
        error_data = {
            'error': True,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'status_code': status_code
        }
        self._send_json_response(error_data, status_code)
    
    def _handle_root(self):
        """Handle root endpoint"""
        response = {
            'service': 'AI Science Platform API',
            'version': '1.0.0',
            'status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'endpoints': {
                'GET /health': 'Health check',
                'GET /status': 'Platform status',
                'GET /api/discover': 'Run discovery with sample data',
                'POST /api/discover': 'Run discovery with provided data',
                'POST /api/model/predict': 'Make model predictions',
                'GET /api/experiments': 'List experiments',
                'POST /api/experiments/run': 'Run experiment'
            }
        }
        self._send_json_response(response)
    
    def _handle_health(self):
        """Handle health check"""
        try:
            # Quick health checks
            engine = DiscoveryEngine()
            runner = ExperimentRunner()
            
            response = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'discovery_engine': 'operational',
                    'experiment_runner': 'operational',
                    'api_server': 'operational'
                },
                'version': '1.0.0'
            }
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            response = {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self._send_json_response(response, 503)
    
    def _handle_status(self):
        """Handle status endpoint"""
        try:
            # Get comprehensive status
            engine = DiscoveryEngine()
            engine_summary = engine.summary()
            
            model = SimpleDiscoveryModel(input_dim=5)
            model_summary = model.get_discovery_summary()
            
            response = {
                'platform_status': 'operational',
                'timestamp': datetime.now().isoformat(),
                'discovery_engine': engine_summary,
                'discovery_model': model_summary,
                'api_version': '1.0.0',
                'capabilities': [
                    'scientific_discovery',
                    'pattern_recognition', 
                    'experiment_management',
                    'model_predictions'
                ]
            }
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            self._send_error(500, f"Status check failed: {e}")
    
    def _handle_discovery(self):
        """Handle discovery with sample data (GET)"""
        try:
            logger.info("Running discovery with sample data")
            
            # Generate sample data
            features, _ = generate_sample_data(size=50, data_type='normal')
            
            # Run discovery
            discoveries = self.discovery_engine.discover(features, context="API_discovery")
            
            response = {
                'discoveries': len(discoveries),
                'timestamp': datetime.now().isoformat(),
                'sample_data_shape': list(features.shape),
                'results': []
            }
            
            for i, discovery in enumerate(discoveries):
                discovery_data = {
                    'id': i + 1,
                    'hypothesis': discovery.hypothesis,
                    'confidence': float(discovery.confidence),
                    'evidence_points': len(discovery.evidence),
                    'metadata': discovery.metadata
                }
                response['results'].append(discovery_data)
            
            logger.info(f"Discovery completed: {len(discoveries)} results")
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            self._send_error(500, f"Discovery failed: {e}")
    
    def _handle_discovery_post(self, data: Dict[str, Any]):
        """Handle discovery with provided data (POST)"""
        try:
            # Extract parameters
            threshold = data.get('threshold', 0.7)
            context = data.get('context', 'API_discovery')
            
            # Handle data input
            if 'data' in data:
                import numpy as np
                features = np.array(data['data'])
            else:
                # Use sample data if none provided
                features, _ = generate_sample_data(size=100, data_type='normal')
            
            logger.info(f"Running discovery on data shape: {features.shape}")
            
            # Create engine with custom threshold
            engine = DiscoveryEngine(discovery_threshold=threshold)
            discoveries = engine.discover(features, context=context)
            
            response = {
                'discoveries': len(discoveries),
                'threshold': threshold,
                'context': context,
                'data_shape': list(features.shape),
                'timestamp': datetime.now().isoformat(),
                'results': []
            }
            
            for i, discovery in enumerate(discoveries):
                discovery_data = {
                    'id': i + 1,
                    'hypothesis': discovery.hypothesis,
                    'confidence': float(discovery.confidence),
                    'evidence_points': len(discovery.evidence),
                    'evidence_summary': discovery.evidence[:3] if len(discovery.evidence) > 0 else [],
                    'metadata': discovery.metadata
                }
                response['results'].append(discovery_data)
            
            logger.info(f"POST discovery completed: {len(discoveries)} results")
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"POST discovery failed: {e}")
            self._send_error(500, f"POST discovery failed: {e}")
    
    def _handle_model_prediction(self, data: Dict[str, Any]):
        """Handle model prediction requests"""
        try:
            import numpy as np
            
            # Extract input data
            if 'input' not in data:
                self._send_error(400, "Missing 'input' field in request")
                return
            
            input_data = np.array(data['input'])
            model_type = data.get('model_type', 'discovery')
            
            logger.info(f"Making {model_type} predictions on shape: {input_data.shape}")
            
            # Create and use model
            if model_type == 'discovery':
                model = SimpleDiscoveryModel(input_dim=input_data.shape[1])
            else:
                from ..models.simple import SimpleModel
                model = SimpleModel(input_dim=input_data.shape[1])
            
            predictions = model.predict(input_data)
            
            response = {
                'predictions': predictions.tolist(),
                'input_shape': list(input_data.shape),
                'model_type': model_type,
                'model_info': model.get_model_info(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Predictions completed: {len(predictions)} results")
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            self._send_error(500, f"Model prediction failed: {e}")
    
    def _handle_list_experiments(self):
        """Handle list experiments"""
        try:
            experiments = self.experiment_runner.list_experiments()
            
            response = {
                'experiments': len(experiments),
                'timestamp': datetime.now().isoformat(),
                'available_experiments': experiments
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"List experiments failed: {e}")
            self._send_error(500, f"List experiments failed: {e}")
    
    def _handle_run_experiment(self, data: Dict[str, Any]):
        """Handle run experiment requests"""
        try:
            name = data.get('name', 'API_experiment')
            runs = data.get('runs', 3)
            
            logger.info(f"Running experiment: {name} with {runs} runs")
            
            # Simple experiment function
            def experiment_function(params):
                engine = DiscoveryEngine(discovery_threshold=0.7)
                discoveries = engine.discover(params["data"], context=name)
                
                return {
                    "discoveries": len(discoveries),
                    "confidence": np.mean([d.confidence for d in discoveries]) if discoveries else 0.0
                }
            
            # Generate sample data for experiment
            features, _ = generate_sample_data(size=100, data_type='normal')
            
            # Run experiment (simplified)
            results = []
            for run in range(runs):
                try:
                    result = experiment_function({"data": features})
                    results.append(result)
                except Exception as e:
                    logger.error(f"Experiment run {run} failed: {e}")
            
            response = {
                'experiment_name': name,
                'runs_completed': len(results),
                'runs_requested': runs,
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'summary': {
                    'avg_discoveries': np.mean([r['discoveries'] for r in results]) if results else 0,
                    'avg_confidence': np.mean([r['confidence'] for r in results]) if results else 0
                }
            }
            
            logger.info(f"Experiment completed: {len(results)} successful runs")
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"Run experiment failed: {e}")
            self._send_error(500, f"Run experiment failed: {e}")
    
    def log_message(self, format, *args):
        """Override log message to use our logger"""
        logger.info(f"{self.address_string()} - {format % args}")


def create_app():
    """Create AI Science Platform API application"""
    class APIServer:
        def __init__(self):
            self.handler = AIScientificPlatformHandler
        
        def run(self, host='0.0.0.0', port=8000, debug=False):
            """Run the API server"""
            setup_logging()
            logger.info(f"Starting AI Science Platform API server on {host}:{port}")
            
            try:
                with socketserver.TCPServer((host, port), self.handler) as httpd:
                    logger.info(f"API server running at http://{host}:{port}")
                    logger.info("Available endpoints:")
                    logger.info("  GET  / - API information")
                    logger.info("  GET  /health - Health check")
                    logger.info("  GET  /status - Platform status")
                    logger.info("  GET  /api/discover - Discovery with sample data")
                    logger.info("  POST /api/discover - Discovery with custom data")
                    logger.info("  POST /api/model/predict - Model predictions")
                    logger.info("  GET  /api/experiments - List experiments")
                    logger.info("  POST /api/experiments/run - Run experiment")
                    
                    httpd.serve_forever()
            except KeyboardInterrupt:
                logger.info("API server shutting down...")
            except Exception as e:
                logger.error(f"API server error: {e}")
                raise
    
    return APIServer()


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)