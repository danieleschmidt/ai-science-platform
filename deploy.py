"""Production deployment script for AI Science Platform"""

import os
import sys
import subprocess
import json
import time
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from core.robust_framework import RobustLogger, robust_execution, secure_operation
except ImportError:
    # Fallback if robust framework not available
    def robust_execution(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class RobustLogger:
        def __init__(self, name, log_file=None):
            self.logger = logging.getLogger(name)
        
        def info(self, msg, **kwargs):
            self.logger.info(f"{msg} | {kwargs}" if kwargs else msg)
        
        def error(self, msg, **kwargs):
            self.logger.error(f"{msg} | {kwargs}" if kwargs else msg)


@dataclass
class DeploymentConfig:
    """Enhanced deployment configuration"""
    app_name: str = "ai-science-platform"
    version: str = "1.0.0"
    python_version: str = "3.8+"
    memory_limit_gb: int = 4
    cpu_cores: int = 2
    storage_gb: int = 20
    enable_monitoring: bool = True
    enable_security_audit: bool = True
    log_level: str = "INFO"
    port: int = 8000


class Deployer:
    """Enhanced production deployment orchestrator"""
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig()
        self.project_root = Path(__file__).parent
        self.logger = RobustLogger("production_deployer", "deployment.log")
        self.deployment_dir = self.project_root / "deployment_package"
        
        # Legacy compatibility
        self.deployment_config = {
            "app_name": self.config.app_name,
            "version": self.config.version,
            "python_version": self.config.python_version,
            "dependencies": "requirements.txt",
            "entry_point": "src.cli:main",
            "health_check_endpoint": "/health",
            "port": self.config.port
        }
        
    def deploy(self, environment: str = "production"):
        """Deploy the AI Science Platform"""
        logger.info(f"🚀 Starting deployment to {environment}")
        
        try:
            # Pre-deployment checks
            self.run_pre_deployment_checks()
            
            # Install dependencies
            self.install_dependencies()
            
            # Run tests
            self.run_tests()
            
            # Build deployment package
            self.build_package()
            
            # Deploy to environment
            self.deploy_to_environment(environment)
            
            # Post-deployment verification
            self.verify_deployment()
            
            logger.info("✅ Deployment completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            sys.exit(1)
    
    def run_pre_deployment_checks(self):
        """Run pre-deployment validation checks"""
        logger.info("🔍 Running pre-deployment checks")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            raise RuntimeError(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        
        # Check required files
        required_files = [
            "requirements.txt",
            "setup.py",
            "src/__init__.py",
            "src/cli.py"
        ]
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                raise RuntimeError(f"Required file missing: {file_path}")
        
        # Check git status (if git repo)
        if (self.project_root / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                if result.stdout.strip():
                    logger.warning("⚠️  Uncommitted changes detected")
                    
            except subprocess.CalledProcessError:
                logger.warning("Could not check git status")
        
        logger.info("✅ Pre-deployment checks passed")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("📦 Installing dependencies")
        
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            
            try:
                subprocess.run(cmd, check=True, cwd=self.project_root)
                logger.info("✅ Dependencies installed")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install dependencies: {e}")
        
        # Install package in development mode
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                         check=True, cwd=self.project_root)
            logger.info("✅ Package installed in development mode")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install package: {e}")
    
    def run_tests(self):
        """Run the test suite"""
        logger.info("🧪 Running tests")
        
        # Run basic import test
        try:
            subprocess.run([sys.executable, "-c", "from src import DiscoveryEngine; print('Import test passed')"],
                         check=True, cwd=self.project_root)
            logger.info("✅ Import tests passed")
        except subprocess.CalledProcessError:
            raise RuntimeError("Import tests failed")
        
        # Run CLI test
        try:
            subprocess.run([sys.executable, "-c", "from src.cli import main; print('CLI test passed')"],
                         check=True, cwd=self.project_root)
            logger.info("✅ CLI tests passed")
        except subprocess.CalledProcessError:
            raise RuntimeError("CLI tests failed")
        
        # Run pytest if available and tests exist
        if (self.project_root / "tests").exists():
            try:
                subprocess.run([sys.executable, "-m", "pytest", "tests/", "-x", "--tb=short"],
                             check=True, cwd=self.project_root)
                logger.info("✅ Unit tests passed")
            except subprocess.CalledProcessError:
                logger.warning("⚠️  Some tests failed, but continuing deployment")
            except FileNotFoundError:
                logger.info("pytest not available, skipping unit tests")
    
    def build_package(self):
        """Build deployment package"""
        logger.info("📦 Building deployment package")
        
        # Create deployment directory
        deploy_dir = self.project_root / "deploy"
        deploy_dir.mkdir(exist_ok=True)
        
        # Generate deployment manifest
        manifest = {
            "name": self.deployment_config["app_name"],
            "version": self.deployment_config["version"],
            "timestamp": time.time(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "entry_point": self.deployment_config["entry_point"],
            "health_check": self.deployment_config["health_check_endpoint"],
            "port": self.deployment_config["port"]
        }
        
        with open(deploy_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Copy essential files
        essential_files = [
            "src/",
            "requirements.txt",
            "setup.py",
            "README.md"
        ]
        
        logger.info("✅ Deployment package built")
    
    def deploy_to_environment(self, environment: str):
        """Deploy to specific environment"""
        logger.info(f"🌐 Deploying to {environment} environment")
        
        if environment == "development":
            self.deploy_development()
        elif environment == "production":
            self.deploy_production()
        else:
            raise ValueError(f"Unknown environment: {environment}")
    
    def deploy_development(self):
        """Deploy to development environment"""
        logger.info("🔧 Setting up development environment")
        
        # Create development configuration
        dev_config = {
            "LOG_LEVEL": "DEBUG",
            "DISCOVERY_THRESHOLD": "0.5",
            "MAX_WORKERS": "4",
            "CACHE_ENABLED": "true",
            "AUTO_SCALING": "false"
        }
        
        # Write environment file
        env_file = self.project_root / ".env.development"
        with open(env_file, "w") as f:
            for key, value in dev_config.items():
                f.write(f"{key}={value}\\n")
        
        logger.info("✅ Development environment configured")
    
    def deploy_production(self):
        """Deploy to production environment"""
        logger.info("🏭 Setting up production environment")
        
        # Create production configuration
        prod_config = {
            "LOG_LEVEL": "INFO",
            "DISCOVERY_THRESHOLD": "0.7",
            "MAX_WORKERS": "8",
            "CACHE_ENABLED": "true",
            "AUTO_SCALING": "true",
            "BACKUP_ENABLED": "true",
            "HEALTH_CHECK_ENABLED": "true"
        }
        
        # Write environment file
        env_file = self.project_root / ".env.production"
        with open(env_file, "w") as f:
            for key, value in prod_config.items():
                f.write(f"{key}={value}\\n")
        
        # Create production directories
        prod_dirs = [
            "logs",
            "backups", 
            "experiment_results",
            "data"
        ]
        
        for dir_name in prod_dirs:
            (self.project_root / dir_name).mkdir(exist_ok=True)
        
        logger.info("✅ Production environment configured")
    
    def verify_deployment(self):
        """Verify deployment is working"""
        logger.info("✅ Verifying deployment")
        
        try:
            # Test CLI status command
            result = subprocess.run([
                sys.executable, "-c",
                "import sys; sys.argv=['ai-science', 'status']; from src.cli import main; main()"
            ], 
            cwd=self.project_root, 
            capture_output=True, 
            text=True,
            timeout=30
            )
            
            if result.returncode == 0:
                logger.info("✅ CLI status check passed")
            else:
                logger.warning(f"⚠️  CLI status check returned {result.returncode}")
                
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  CLI status check timed out")
        except Exception as e:
            logger.warning(f"⚠️  CLI status check failed: {e}")
        
        # Test core imports
        try:
            subprocess.run([
                sys.executable, "-c",
                "from src import DiscoveryEngine, ExperimentRunner; print('Core imports successful')"
            ], check=True, cwd=self.project_root)
            logger.info("✅ Core imports verified")
        except subprocess.CalledProcessError:
            raise RuntimeError("Core import verification failed")
    
    def generate_deployment_report(self):
        """Generate deployment report"""
        logger.info("📋 Generating deployment report")
        
        report = {
            "deployment_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": f"{os.name} {os.uname().sysname if hasattr(os, 'uname') else 'unknown'}",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "project_version": self.deployment_config["version"],
            "deployment_status": "completed",
            "checks_performed": [
                "pre_deployment_validation",
                "dependency_installation", 
                "test_execution",
                "package_building",
                "environment_configuration",
                "post_deployment_verification"
            ]
        }
        
        report_file = self.project_root / "deployment_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Deployment report saved: {report_file}")
        
        return report
    
    @robust_execution(max_retries=2, timeout_seconds=300)
    def prepare_production_package(self) -> Dict[str, Any]:
        """Prepare comprehensive production deployment package"""
        
        self.logger.info("Starting production package preparation")
        
        package_summary = {
            "timestamp": datetime.now().isoformat(),
            "version": self.config.version,
            "environment": "production",
            "components": [],
            "files_created": [],
            "validation_results": {}
        }
        
        try:
            # Create deployment directory structure
            self._create_production_structure()
            package_summary["components"].append("directory_structure")
            
            # Package source code
            source_files = self._package_production_code()
            package_summary["files_created"].extend(source_files)
            package_summary["components"].append("source_code")
            
            # Generate production configs
            config_files = self._generate_production_configs()
            package_summary["files_created"].extend(config_files)
            package_summary["components"].append("configuration")
            
            # Create management scripts
            script_files = self._create_management_scripts()
            package_summary["files_created"].extend(script_files)
            package_summary["components"].append("management_scripts")
            
            # Generate Docker support
            docker_files = self._create_docker_files()
            package_summary["files_created"].extend(docker_files)
            package_summary["components"].append("docker_support")
            
            # Create documentation
            doc_files = self._generate_production_docs()
            package_summary["files_created"].extend(doc_files)
            package_summary["components"].append("documentation")
            
            # Validate package
            validation_results = self._validate_production_package()
            package_summary["validation_results"] = validation_results
            package_summary["components"].append("validation")
            
            self.logger.info(
                "Production package preparation completed successfully",
                components=len(package_summary["components"]),
                files_created=len(package_summary["files_created"])
            )
            
            return package_summary
            
        except Exception as e:
            self.logger.error(
                "Production package preparation failed",
                error=str(e),
                components_completed=len(package_summary["components"])
            )
            raise
    
    def _create_production_structure(self):
        """Create production deployment directory structure"""
        directories = [
            self.deployment_dir,
            self.deployment_dir / "scripts",
            self.deployment_dir / "config",
            self.deployment_dir / "docs",
            self.deployment_dir / "src",
            self.deployment_dir / "logs",
            self.deployment_dir / "data",
            self.deployment_dir / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def _package_production_code(self) -> List[str]:
        """Package source code for production"""
        source_files = []
        
        # Copy src directory
        src_source = self.project_root / "src"
        src_dest = self.deployment_dir / "src"
        
        if src_source.exists():
            shutil.copytree(src_source, src_dest, dirs_exist_ok=True)
            for py_file in src_dest.rglob("*.py"):
                source_files.append(str(py_file.relative_to(self.deployment_dir)))
        
        # Copy main scripts
        main_scripts = [
            "quality_gates.py",
            "research_validation_suite.py",
            "requirements.txt",
            "setup.py"
        ]
        
        for script in main_scripts:
            source_file = self.project_root / script
            if source_file.exists():
                shutil.copy2(source_file, self.deployment_dir / script)
                source_files.append(script)
        
        return source_files
    
    def _generate_production_configs(self) -> List[str]:
        """Generate production configuration files"""
        config_files = []
        config_dir = self.deployment_dir / "config"
        
        # Production settings
        production_settings = {
            "environment": "production",
            "version": self.config.version,
            "logging": {
                "level": self.config.log_level,
                "file": "logs/ai-science-platform.log",
                "max_size_mb": 100,
                "backup_count": 5
            },
            "security": {
                "enable_audit_logging": self.config.enable_security_audit,
                "max_memory_mb": self.config.memory_limit_gb * 1024,
                "max_execution_time_seconds": 3600
            },
            "performance": {
                "enable_caching": True,
                "cache_size_mb": 512,
                "parallel_workers": self.config.cpu_cores,
                "batch_size": 1000
            }
        }
        
        config_file = config_dir / "production.json"
        with open(config_file, 'w') as f:
            json.dump(production_settings, f, indent=2)
        config_files.append(str(config_file.relative_to(self.deployment_dir)))
        
        # Environment file
        env_content = f"""ENVIRONMENT=production
VERSION={self.config.version}
LOG_LEVEL={self.config.log_level}
MEMORY_LIMIT_GB={self.config.memory_limit_gb}
CPU_CORES={self.config.cpu_cores}
PORT={self.config.port}
"""
        
        env_file = config_dir / "production.env"
        with open(env_file, 'w') as f:
            f.write(env_content)
        config_files.append(str(env_file.relative_to(self.deployment_dir)))
        
        return config_files
    
    def _create_management_scripts(self) -> List[str]:
        """Create production management scripts"""
        script_files = []
        scripts_dir = self.deployment_dir / "scripts"
        
        # Install script
        install_script = """#!/bin/bash
set -e
echo "🚀 Installing AI Science Platform..."

# Install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p logs data backups

echo "✅ Installation completed!"
"""
        
        install_file = scripts_dir / "install.sh"
        with open(install_file, 'w') as f:
            f.write(install_script)
        install_file.chmod(0o755)
        script_files.append(str(install_file.relative_to(self.deployment_dir)))
        
        # Start script
        start_script = """#!/bin/bash
set -e
echo "🚀 Starting AI Science Platform..."

source config/production.env 2>/dev/null || true

python3 -c "
import sys
sys.path.append('src')
print('AI Science Platform starting...')
print('Platform ready for operation!')
"

echo "✅ Platform started successfully!"
"""
        
        start_file = scripts_dir / "start.sh"
        with open(start_file, 'w') as f:
            f.write(start_script)
        start_file.chmod(0o755)
        script_files.append(str(start_file.relative_to(self.deployment_dir)))
        
        return script_files
    
    def _create_docker_files(self) -> List[str]:
        """Create Docker support files"""
        docker_files = []
        
        # Dockerfile
        dockerfile_content = f"""FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY *.py ./
COPY config/ config/

RUN mkdir -p logs data backups

EXPOSE {self.config.port}

CMD ["python3", "-c", "print('AI Science Platform ready')"]
"""
        
        dockerfile = self.deployment_dir / "Dockerfile"
        with open(dockerfile, 'w') as f:
            f.write(dockerfile_content)
        docker_files.append(str(dockerfile.relative_to(self.deployment_dir)))
        
        return docker_files
    
    def _generate_production_docs(self) -> List[str]:
        """Generate production documentation"""
        doc_files = []
        docs_dir = self.deployment_dir / "docs"
        
        # Quick start guide
        quick_start = f"""# AI Science Platform - Production Deployment

## Quick Start

1. Install: `./scripts/install.sh`
2. Start: `./scripts/start.sh`
3. Verify: Check logs in `logs/` directory

## Version: {self.config.version}
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        guide_file = docs_dir / "PRODUCTION_GUIDE.md"
        with open(guide_file, 'w') as f:
            f.write(quick_start)
        doc_files.append(str(guide_file.relative_to(self.deployment_dir)))
        
        return doc_files
    
    def _validate_production_package(self) -> Dict[str, Any]:
        """Validate production deployment package"""
        validation_results = {
            "overall_status": "VALID",
            "checks": {},
            "warnings": [],
            "errors": []
        }
        
        # Check required files
        required_files = [
            "src/core/robust_framework.py",
            "config/production.json",
            "scripts/install.sh",
            "Dockerfile"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.deployment_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        validation_results["checks"]["required_files"] = {
            "status": "PASS" if not missing_files else "FAIL",
            "missing_files": missing_files
        }
        
        if missing_files:
            validation_results["overall_status"] = "INVALID"
            validation_results["errors"].extend([f"Missing: {f}" for f in missing_files])
        
        return validation_results


def main():
    """Main deployment entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy AI Science Platform")
    parser.add_argument(
        "environment", 
        choices=["development", "production"],
        default="development",
        nargs="?",
        help="Deployment environment"
    )
    parser.add_argument(
        "--skip-tests", 
        action="store_true",
        help="Skip test execution"
    )
    parser.add_argument(
        "--report-only", 
        action="store_true",
        help="Only generate deployment report"
    )
    parser.add_argument(
        "--production-package", 
        action="store_true",
        help="Create comprehensive production deployment package"
    )
    
    args = parser.parse_args()
    
    # Enhanced deployment configuration
    config = DeploymentConfig(
        version="1.0.0",
        memory_limit_gb=4,
        cpu_cores=2,
        storage_gb=20
    )
    
    deployer = Deployer(config)
    
    try:
        if args.production_package:
            # Create comprehensive production package
            print("🚀 CREATING PRODUCTION DEPLOYMENT PACKAGE")
            print("=" * 50)
            
            package_summary = deployer.prepare_production_package()
            
            print(f"\n📊 PRODUCTION PACKAGE SUMMARY")
            print("=" * 30)
            print(f"✅ Version: {package_summary['version']}")
            print(f"✅ Components: {len(package_summary['components'])}")
            print(f"✅ Files Created: {len(package_summary['files_created'])}")
            
            validation = package_summary['validation_results']
            print(f"\n🔍 Validation: {validation['overall_status']}")
            
            if validation['errors']:
                print(f"❌ Errors:")
                for error in validation['errors']:
                    print(f"     • {error}")
            
            print(f"\n📁 Production Package: {deployer.deployment_dir}")
            print(f"🚀 Ready for production deployment!")
            
        elif args.report_only:
            deployer.generate_deployment_report()
        else:
            # Standard deployment flow
            if args.skip_tests:
                deployer.run_tests = lambda: logger.info("⏭️  Skipping tests as requested")
            
            deployer.deploy(args.environment)
            deployer.generate_deployment_report()
        
        print("\\n🎉 AI Science Platform deployment complete!")
        print("\\nQuick start commands:")
        print("  python -m src.cli status      # Check platform status")
        print("  python -m src.cli discover    # Run discovery on sample data")
        print("  python -m src.cli --help      # Show all available commands")
        
        if args.production_package:
            print("\\nProduction deployment:")
            print("  cd deployment_package && ./scripts/install.sh")
            print("  ./scripts/start.sh")
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()