"""Internationalization support for AI Science Platform"""

from .localizer import Localizer, get_localizer, translate, set_language
from .compliance import ComplianceManager, get_compliance_manager
from .cross_platform import CrossPlatformManager, get_platform_manager

__all__ = [
    "Localizer",
    "get_localizer", 
    "translate",
    "set_language",
    "ComplianceManager",
    "get_compliance_manager",
    "CrossPlatformManager",
    "get_platform_manager"
]