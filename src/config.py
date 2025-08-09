"""Configuration management for AI Science Platform"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryConfig:
    """Configuration for discovery engine"""
    discovery_threshold: float = 0.7
    max_hypotheses: int = 10
    min_data_size: int = 10
    confidence_boost_factor: float = 1.2
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class ExperimentConfig:
    """Configuration for experiment runner"""
    default_num_runs: int = 3
    max_num_runs: int = 100
    results_dir: str = "experiment_results"
    enable_parallel: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300


@dataclass
class SecurityConfig:
    """Security configuration"""
    max_string_length: int = 100000
    max_array_elements: int = 10000000
    allowed_file_extensions: list = None
    enable_input_validation: bool = True
    log_security_events: bool = True
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = [".json", ".csv", ".txt", ".log", ".pkl"]


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    log_dir: str = "logs"
    enable_console: bool = True
    enable_file: bool = True
    enable_json: bool = False
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class PlatformConfig:
    """Main platform configuration"""
    discovery: DiscoveryConfig = None
    experiments: ExperimentConfig = None
    security: SecurityConfig = None
    logging: LoggingConfig = None
    environment: str = "development"
    debug: bool = False
    
    def __post_init__(self):
        if self.discovery is None:
            self.discovery = DiscoveryConfig()
        if self.experiments is None:
            self.experiments = ExperimentConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


class ConfigManager:
    """Centralized configuration management"""
    
    DEFAULT_CONFIG_PATHS = [
        "./config.yaml",
        "./config.json",
        "~/.ai_science_platform/config.yaml",
        "/etc/ai_science_platform/config.yaml"
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file or environment"""
        config_data = {}
        
        # Try to load from file
        config_file = self._find_config_file()
        if config_file:
            config_data = self._load_config_file(config_file)
            logger.info(f"Loaded configuration from {config_file}")
        
        # Override with environment variables
        env_overrides = self._load_env_config()
        config_data.update(env_overrides)
        
        # Create configuration object
        self._config = self._create_config_from_dict(config_data)
        
        logger.info("Configuration loaded successfully")
    
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file"""
        search_paths = []
        
        # Use provided path first
        if self.config_path:
            search_paths.append(self.config_path)
        
        # Add default paths
        search_paths.extend(self.DEFAULT_CONFIG_PATHS)
        
        for path_str in search_paths:
            path = Path(path_str).expanduser()
            if path.exists() and path.is_file():
                return path
        
        logger.info("No configuration file found, using defaults")
        return None
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        prefix = "AI_SCIENCE_"
        
        env_mappings = {
            f"{prefix}LOG_LEVEL": ("logging", "log_level"),
            f"{prefix}LOG_DIR": ("logging", "log_dir"),
            f"{prefix}DEBUG": ("debug",),
            f"{prefix}ENVIRONMENT": ("environment",),
            f"{prefix}DISCOVERY_THRESHOLD": ("discovery", "discovery_threshold"),
            f"{prefix}RESULTS_DIR": ("experiments", "results_dir"),
            f"{prefix}MAX_WORKERS": ("experiments", "max_workers"),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if env_var.endswith("_THRESHOLD"):
                    value = float(value)
                elif env_var.endswith(("_WORKERS", "_RUNS", "_SIZE")):
                    value = int(value)
                elif env_var.endswith("_DEBUG"):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                # Set nested configuration
                current = env_config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = value
        
        return env_config
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> PlatformConfig:
        """Create configuration object from dictionary"""
        try:
            # Extract nested configurations
            discovery_data = config_data.get("discovery", {})
            experiments_data = config_data.get("experiments", {})
            security_data = config_data.get("security", {})
            logging_data = config_data.get("logging", {})
            
            # Create configuration objects
            discovery_config = DiscoveryConfig(**discovery_data)
            experiments_config = ExperimentConfig(**experiments_data)
            security_config = SecurityConfig(**security_data)
            logging_config = LoggingConfig(**logging_data)
            
            # Create main configuration
            main_config_data = {
                k: v for k, v in config_data.items() 
                if k not in ["discovery", "experiments", "security", "logging"]
            }
            
            return PlatformConfig(
                discovery=discovery_config,
                experiments=experiments_config,
                security=security_config,
                logging=logging_config,
                **main_config_data
            )
        
        except Exception as e:
            logger.error(f"Failed to create configuration object: {e}")
            # Return default configuration on error
            return PlatformConfig()
    
    def get_config(self) -> PlatformConfig:
        """Get current configuration"""
        return self._config
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        if config_path is None:
            config_path = self.config_path or "./config.yaml"
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = asdict(self._config)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            logger.info(f"Configuration saved to {config_path}")
        
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        try:
            # Convert current config to dict
            config_dict = asdict(self._config)
            
            # Apply updates
            self._deep_update(config_dict, updates)
            
            # Recreate configuration object
            self._config = self._create_config_from_dict(config_dict)
            
            logger.info(f"Configuration updated with: {updates}")
        
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        config = self._config
        
        # Validate discovery configuration
        if not (0.0 <= config.discovery.discovery_threshold <= 1.0):
            validation_results["errors"].append(
                "Discovery threshold must be between 0.0 and 1.0"
            )
            validation_results["valid"] = False
        
        if config.discovery.max_hypotheses < 1:
            validation_results["errors"].append(
                "Max hypotheses must be at least 1"
            )
            validation_results["valid"] = False
        
        # Validate experiment configuration
        if config.experiments.max_num_runs < config.experiments.default_num_runs:
            validation_results["errors"].append(
                "Max num runs must be >= default num runs"
            )
            validation_results["valid"] = False
        
        if config.experiments.max_workers < 1:
            validation_results["errors"].append(
                "Max workers must be at least 1"
            )
            validation_results["valid"] = False
        
        # Validate security configuration
        if config.security.max_string_length < 1000:
            validation_results["warnings"].append(
                "Max string length is very low, may cause issues"
            )
        
        # Validate logging configuration
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.logging.log_level.upper() not in valid_log_levels:
            validation_results["errors"].append(
                f"Invalid log level: {config.logging.log_level}"
            )
            validation_results["valid"] = False
        
        logger.info(f"Configuration validation: {validation_results}")
        return validation_results


# Global configuration manager instance
_config_manager = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get or create global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> PlatformConfig:
    """Get current platform configuration"""
    return get_config_manager().get_config()


def reload_config(config_path: Optional[str] = None) -> None:
    """Reload configuration from file"""
    global _config_manager
    _config_manager = ConfigManager(config_path)