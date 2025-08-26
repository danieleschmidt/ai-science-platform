"""Cross-platform compatibility management"""

import platform
import os
import sys
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PlatformInfo:
    """Platform information"""
    system: str
    release: str
    version: str
    machine: str
    processor: str
    python_version: str
    python_implementation: str
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'system': self.system,
            'release': self.release,
            'version': self.version,
            'machine': self.machine,
            'processor': self.processor,
            'python_version': self.python_version,
            'python_implementation': self.python_implementation
        }


class CrossPlatformManager:
    """Cross-platform compatibility manager"""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        self.supported_platforms = ['Linux', 'Windows', 'Darwin', 'FreeBSD']
        
        logger.info(f"CrossPlatformManager initialized for {self.platform_info.system}")
    
    def _detect_platform(self) -> PlatformInfo:
        """Detect current platform information"""
        return PlatformInfo(
            system=platform.system(),
            release=platform.release(),
            version=platform.version(),
            machine=platform.machine(),
            processor=platform.processor(),
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation()
        )
    
    def is_supported_platform(self) -> bool:
        """Check if current platform is supported"""
        return self.platform_info.system in self.supported_platforms
    
    def get_platform_specific_config(self) -> Dict[str, Any]:
        """Get platform-specific configuration"""
        config = {
            'line_ending': '\n',
            'path_separator': os.sep,
            'env_path_separator': os.pathsep,
            'temp_directory': Path.home() / 'tmp',
            'config_directory': Path.home() / '.ai-science-platform',
            'log_directory': Path.cwd() / 'logs',
            'max_workers': os.cpu_count() or 1
        }
        
        # Platform-specific adjustments
        if self.platform_info.system == 'Windows':
            config.update({
                'line_ending': '\r\n',
                'temp_directory': Path(os.environ.get('TEMP', str(Path.home() / 'AppData/Local/Temp'))),
                'config_directory': Path(os.environ.get('APPDATA', str(Path.home() / 'AppData/Roaming'))) / 'ai-science-platform'
            })
        elif self.platform_info.system == 'Darwin':
            config.update({
                'config_directory': Path.home() / 'Library/Application Support/ai-science-platform',
                'log_directory': Path.home() / 'Library/Logs/ai-science-platform'
            })
        
        return config
    
    def ensure_directories(self) -> bool:
        """Ensure platform-specific directories exist"""
        config = self.get_platform_specific_config()
        
        directories = [
            config['config_directory'],
            config['log_directory'],
            config['temp_directory']
        ]
        
        try:
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get system capabilities and resources"""
        import psutil
        
        return {
            'cpu_count': os.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3),
            'python_bits': platform.architecture()[0],
            'supports_multiprocessing': True,
            'supports_threading': True
        }
    
    def get_compatibility_report(self) -> Dict[str, Any]:
        """Generate platform compatibility report"""
        try:
            capabilities = self.get_system_capabilities()
        except ImportError:
            capabilities = {'cpu_count': os.cpu_count(), 'note': 'psutil not available'}
        
        return {
            'platform_info': self.platform_info.to_dict(),
            'is_supported': self.is_supported_platform(),
            'system_capabilities': capabilities,
            'platform_config': self.get_platform_specific_config(),
            'python_info': {
                'version': sys.version,
                'executable': sys.executable,
                'path': sys.path[:3]  # First 3 path entries
            }
        }


# Global cross-platform manager instance
_platform_manager = None

def get_platform_manager() -> CrossPlatformManager:
    """Get global cross-platform manager instance"""
    global _platform_manager
    if _platform_manager is None:
        _platform_manager = CrossPlatformManager()
        _platform_manager.ensure_directories()
    return _platform_manager


# Example usage
if __name__ == "__main__":
    manager = CrossPlatformManager()
    
    print("Platform Compatibility Report:")
    report = manager.get_compatibility_report()
    
    print(f"System: {report['platform_info']['system']} {report['platform_info']['release']}")
    print(f"Supported: {report['is_supported']}")
    print(f"CPU Count: {report['system_capabilities'].get('cpu_count', 'Unknown')}")
    print(f"Python: {report['platform_info']['python_version']} ({report['platform_info']['python_implementation']})")