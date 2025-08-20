"""Main entry point for AI Science Platform"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.health_check import get_health_checker
from src.config import get_config_manager


def setup_environment():
    """Setup the runtime environment"""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting AI Science Platform")
    
    # Check health
    health_checker = get_health_checker()
    health_status = health_checker.run_health_check()
    
    if health_status['status'] != 'HEALTHY':
        logger.error(f"Health check failed: {health_status}")
        return False
    
    logger.info("Health check passed")
    return True


def run_api_server():
    """Run the API server"""
    try:
        from src.api.server import create_app
        
        app = create_app()
        port = int(os.getenv('PORT', 8000))
        workers = int(os.getenv('WORKERS', 1))
        
        logger = logging.getLogger(__name__)
        logger.info(f"Starting API server on port {port} with {workers} workers")
        
        # In production, you would use a proper WSGI server like gunicorn
        # For now, we'll use the development server
        if os.getenv('ENVIRONMENT') == 'development':
            app.run(host='0.0.0.0', port=port, debug=True)
        else:
            # Production would use gunicorn or similar
            app.run(host='0.0.0.0', port=port, debug=False)
            
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("API server not available, running in basic mode")
        return False
    
    return True


def run_basic_mode():
    """Run in basic demonstration mode"""
    logger = logging.getLogger(__name__)
    logger.info("Running in basic demonstration mode")
    
    try:
        # Import core modules
        from src.models.simple import SimpleModel, SimpleDiscoveryModel
        from src.algorithms.discovery import DiscoveryEngine
        from src.utils.data_utils import generate_sample_data
        
        # Create a simple demo
        logger.info("Creating AI Science Platform demo")
        
        # Generate sample data
        features, targets = generate_sample_data(size=100, data_type='normal')
        logger.info(f"Generated sample data: {features.shape}")
        
        # Create and test model
        model = SimpleDiscoveryModel(input_dim=1, hidden_dim=32)
        logger.info(f"Created discovery model: {model.get_model_info()}")
        
        # Test discovery
        engine = DiscoveryEngine()
        discoveries = engine.discover(features)
        logger.info(f"Discovery engine found: {len(discoveries) if isinstance(discoveries, list) else 'error'}")
        
        # Keep running
        import time
        logger.info("Demo completed successfully. Platform is running...")
        
        while True:
            time.sleep(60)
            logger.info("AI Science Platform is active")
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in basic mode: {e}")
        return False
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI Science Platform')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    parser.add_argument('--api', action='store_true', help='Run API server mode')
    parser.add_argument('--basic', action='store_true', help='Run in basic mode')
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Determine run mode
        if args.api:
            logger.info("Starting in API server mode")
            success = run_api_server()
        elif args.basic:
            logger.info("Starting in basic mode")
            success = run_basic_mode()
        elif args.dev:
            logger.info("Starting in development mode")
            success = run_api_server() or run_basic_mode()
        else:
            # Default mode
            logger.info("Starting in default mode")
            success = run_api_server() if os.getenv('ENABLE_API', 'false').lower() == 'true' else run_basic_mode()
        
        if not success:
            logger.error("Failed to start platform")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down AI Science Platform")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()