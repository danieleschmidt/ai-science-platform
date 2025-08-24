"""Advanced Security Framework for Research Operations"""

import hashlib
import hmac
import secrets
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import os
from pathlib import Path

from ..utils.error_handling import robust_execution, SecurityError
from ..utils.validation import ValidationMixin

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class OperationType(Enum):
    """Types of research operations"""
    DATA_ACCESS = "data_access"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EXPERIMENT_DESIGN = "experiment_design"
    RESULT_VALIDATION = "result_validation"
    MODEL_TRAINING = "model_training"
    CAUSAL_DISCOVERY = "causal_discovery"
    PUBLICATION_PREP = "publication_prep"


@dataclass
class SecurityCredential:
    """Security credentials for research operations"""
    user_id: str
    role: str
    permissions: List[str]
    security_clearance: SecurityLevel
    issued_at: datetime
    expires_at: datetime
    access_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    
    def is_valid(self) -> bool:
        """Check if credential is still valid"""
        return datetime.now() < self.expires_at
    
    def has_permission(self, operation: str) -> bool:
        """Check if credential has specific permission"""
        return operation in self.permissions or "admin" in self.permissions


@dataclass
class SecurityAuditLog:
    """Security audit log entry"""
    timestamp: datetime
    user_id: str
    operation: OperationType
    resource: str
    result: str  # success, failure, blocked
    security_level: SecurityLevel
    ip_address: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatDetection:
    """Threat detection result"""
    threat_id: str
    threat_type: str
    severity: str  # low, medium, high, critical
    description: str
    indicators: List[str]
    recommended_actions: List[str]
    confidence_score: float
    detected_at: datetime = field(default_factory=datetime.now)


class ResearchSecurityManager(ValidationMixin):
    """Advanced security manager for research operations"""
    
    def __init__(self, 
                 secret_key: str = None,
                 audit_enabled: bool = True,
                 threat_detection_enabled: bool = True):
        """
        Initialize research security manager
        
        Args:
            secret_key: Secret key for cryptographic operations
            audit_enabled: Enable security audit logging
            threat_detection_enabled: Enable threat detection
        """
        self.secret_key = secret_key or self._generate_secret_key()
        self.audit_enabled = audit_enabled
        self.threat_detection_enabled = threat_detection_enabled
        
        # Security state
        self.active_credentials: Dict[str, SecurityCredential] = {}
        self.security_audit_log: List[SecurityAuditLog] = []
        self.detected_threats: List[ThreatDetection] = []
        
        # Security policies
        self.security_policies = self._initialize_security_policies()
        
        # Rate limiting
        self.rate_limits: Dict[str, List[datetime]] = {}
        
        logger.info("ResearchSecurityManager initialized with advanced protection")
    
    @robust_execution(recovery_strategy='security_fallback')
    def authenticate_user(self, 
                         user_id: str, 
                         credentials: Dict[str, Any],
                         requested_permissions: List[str] = None) -> SecurityCredential:
        """
        Authenticate user and generate security credential
        
        Args:
            user_id: User identifier
            credentials: Authentication credentials
            requested_permissions: Requested permissions
            
        Returns:
            SecurityCredential for authenticated user
        """
        
        # Validate credentials
        if not self._validate_credentials(user_id, credentials):
            self._log_security_event(user_id, OperationType.DATA_ACCESS, "authentication_failed", 
                                    SecurityLevel.PUBLIC, details={"reason": "invalid_credentials"})
            raise SecurityError("Authentication failed")
        
        # Check for threats
        if self.threat_detection_enabled:
            threats = self._detect_authentication_threats(user_id, credentials)
            if threats:
                self._log_security_event(user_id, OperationType.DATA_ACCESS, "threat_detected", 
                                       SecurityLevel.RESTRICTED, details={"threats": len(threats)})
                raise SecurityError(f"Security threats detected: {len(threats)} threats")
        
        # Determine security clearance
        security_clearance = self._determine_security_clearance(user_id, credentials)
        
        # Generate permissions
        permissions = self._generate_permissions(user_id, security_clearance, requested_permissions)
        
        # Create credential
        credential = SecurityCredential(
            user_id=user_id,
            role=credentials.get('role', 'researcher'),
            permissions=permissions,
            security_clearance=security_clearance,
            issued_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
        # Store active credential
        self.active_credentials[credential.access_token] = credential
        
        # Log successful authentication
        self._log_security_event(user_id, OperationType.DATA_ACCESS, "authentication_success", 
                               security_clearance, details={"permissions": len(permissions)})
        
        logger.info(f"User {user_id} authenticated with {security_clearance.value} clearance")
        return credential
    
    def authorize_operation(self, 
                          access_token: str,
                          operation: OperationType,
                          resource: str,
                          security_level: SecurityLevel = SecurityLevel.INTERNAL) -> bool:
        """
        Authorize specific research operation
        
        Args:
            access_token: User access token
            operation: Type of operation
            resource: Resource being accessed
            security_level: Required security level
            
        Returns:
            True if authorized, False otherwise
        """
        
        # Validate token
        credential = self.active_credentials.get(access_token)
        if not credential or not credential.is_valid():
            self._log_security_event("unknown", operation, "authorization_failed", 
                                   security_level, details={"reason": "invalid_token"})
            return False
        
        # Check rate limits
        if not self._check_rate_limit(credential.user_id, operation):
            self._log_security_event(credential.user_id, operation, "rate_limit_exceeded", 
                                   security_level, details={"resource": resource})
            return False
        
        # Check security clearance
        if not self._check_security_clearance(credential.security_clearance, security_level):
            self._log_security_event(credential.user_id, operation, "clearance_insufficient", 
                                   security_level, details={"required": security_level.value})
            return False
        
        # Check specific permissions
        operation_permission = self._get_operation_permission(operation)
        if not credential.has_permission(operation_permission):
            self._log_security_event(credential.user_id, operation, "permission_denied", 
                                   security_level, details={"required_permission": operation_permission})
            return False
        
        # Check resource-specific policies
        if not self._check_resource_policies(credential, operation, resource, security_level):
            self._log_security_event(credential.user_id, operation, "policy_violation", 
                                   security_level, details={"resource": resource})
            return False
        
        # Log successful authorization
        self._log_security_event(credential.user_id, operation, "authorized", 
                               security_level, details={"resource": resource})
        
        return True
    
    def secure_data_encryption(self, data: bytes, security_level: SecurityLevel) -> Tuple[bytes, str]:
        """
        Encrypt sensitive data based on security level
        
        Args:
            data: Data to encrypt
            security_level: Required security level
            
        Returns:
            Tuple of (encrypted_data, encryption_key_id)
        """
        
        # Generate encryption key based on security level
        key_size = self._get_key_size_for_level(security_level)
        encryption_key = secrets.token_bytes(key_size // 8)
        
        # Simple XOR encryption (in production, use proper AES/ChaCha20)
        encrypted_data = bytearray()
        for i, byte in enumerate(data):
            encrypted_data.append(byte ^ encryption_key[i % len(encryption_key)])
        
        # Generate key ID
        key_id = hashlib.sha256(encryption_key).hexdigest()[:16]
        
        # Store key securely (in production, use HSM/key vault)
        self._store_encryption_key(key_id, encryption_key, security_level)
        
        logger.info(f"Data encrypted with {security_level.value} security level")
        return bytes(encrypted_data), key_id
    
    def secure_data_decryption(self, encrypted_data: bytes, key_id: str) -> bytes:
        """
        Decrypt data using stored key
        
        Args:
            encrypted_data: Encrypted data
            key_id: Encryption key identifier
            
        Returns:
            Decrypted data
        """
        
        # Retrieve encryption key
        encryption_key = self._retrieve_encryption_key(key_id)
        if not encryption_key:
            raise SecurityError(f"Encryption key not found: {key_id}")
        
        # Decrypt (reverse XOR)
        decrypted_data = bytearray()
        for i, byte in enumerate(encrypted_data):
            decrypted_data.append(byte ^ encryption_key[i % len(encryption_key)])
        
        return bytes(decrypted_data)
    
    def detect_security_threats(self, 
                               operation_data: Dict[str, Any],
                               context: Dict[str, Any] = None) -> List[ThreatDetection]:
        """
        Detect security threats in research operations
        
        Args:
            operation_data: Data from research operation
            context: Additional context for threat detection
            
        Returns:
            List of detected threats
        """
        
        if not self.threat_detection_enabled:
            return []
        
        threats = []
        
        # Data exfiltration detection
        data_threat = self._detect_data_exfiltration(operation_data)
        if data_threat:
            threats.append(data_threat)
        
        # Unusual access patterns
        access_threat = self._detect_unusual_access_patterns(operation_data, context)
        if access_threat:
            threats.append(access_threat)
        
        # Malicious input detection
        input_threat = self._detect_malicious_inputs(operation_data)
        if input_threat:
            threats.append(input_threat)
        
        # Privacy violations
        privacy_threat = self._detect_privacy_violations(operation_data)
        if privacy_threat:
            threats.append(privacy_threat)
        
        # Store detected threats
        self.detected_threats.extend(threats)
        
        if threats:
            logger.warning(f"Detected {len(threats)} security threats")
        
        return threats
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        
        # Analyze audit logs
        recent_logs = [log for log in self.security_audit_log 
                      if log.timestamp > datetime.now() - timedelta(days=30)]
        
        # Count operations by result
        success_count = sum(1 for log in recent_logs if log.result == "success")
        failure_count = sum(1 for log in recent_logs if log.result == "failure") 
        blocked_count = sum(1 for log in recent_logs if log.result == "blocked")
        
        # Analyze threats
        recent_threats = [threat for threat in self.detected_threats
                         if threat.detected_at > datetime.now() - timedelta(days=30)]
        
        threat_by_severity = {}
        for threat in recent_threats:
            severity = threat.severity
            threat_by_severity[severity] = threat_by_severity.get(severity, 0) + 1
        
        # Active sessions
        active_sessions = len([cred for cred in self.active_credentials.values() 
                              if cred.is_valid()])
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "reporting_period": "30_days",
            "access_statistics": {
                "total_operations": len(recent_logs),
                "successful_operations": success_count,
                "failed_operations": failure_count,
                "blocked_operations": blocked_count,
                "success_rate": success_count / max(1, len(recent_logs))
            },
            "threat_analysis": {
                "total_threats_detected": len(recent_threats),
                "threats_by_severity": threat_by_severity,
                "threat_detection_rate": len(recent_threats) / max(1, len(recent_logs))
            },
            "security_metrics": {
                "active_sessions": active_sessions,
                "encryption_operations": sum(1 for log in recent_logs 
                                           if "encrypt" in log.details.get("operation", "")),
                "high_security_operations": sum(1 for log in recent_logs
                                              if log.security_level in [SecurityLevel.RESTRICTED, SecurityLevel.TOP_SECRET])
            },
            "recommendations": self._generate_security_recommendations(recent_logs, recent_threats)
        }
        
        return report
    
    def _generate_secret_key(self) -> str:
        """Generate secure secret key"""
        return secrets.token_urlsafe(32)
    
    def _initialize_security_policies(self) -> Dict[str, Any]:
        """Initialize security policies"""
        return {
            "max_failed_attempts": 5,
            "session_timeout_hours": 24,
            "rate_limits": {
                OperationType.DATA_ACCESS: {"requests": 1000, "window_minutes": 60},
                OperationType.MODEL_TRAINING: {"requests": 10, "window_minutes": 60},
                OperationType.CAUSAL_DISCOVERY: {"requests": 50, "window_minutes": 60}
            },
            "encryption_requirements": {
                SecurityLevel.CONFIDENTIAL: 256,
                SecurityLevel.RESTRICTED: 256,
                SecurityLevel.TOP_SECRET: 512
            }
        }
    
    def _validate_credentials(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Validate user credentials"""
        # In production, this would check against a proper identity provider
        required_fields = ['password', 'role']
        
        for field in required_fields:
            if field not in credentials:
                return False
        
        # Simple validation (in production, use proper hashing/OAuth)
        if len(credentials['password']) < 8:
            return False
        
        return True
    
    def _determine_security_clearance(self, user_id: str, credentials: Dict[str, Any]) -> SecurityLevel:
        """Determine user security clearance"""
        role = credentials.get('role', 'researcher')
        
        clearance_mapping = {
            'admin': SecurityLevel.RESTRICTED,
            'senior_researcher': SecurityLevel.CONFIDENTIAL,
            'researcher': SecurityLevel.INTERNAL,
            'intern': SecurityLevel.PUBLIC
        }
        
        return clearance_mapping.get(role, SecurityLevel.PUBLIC)
    
    def _generate_permissions(self, user_id: str, clearance: SecurityLevel, requested: List[str] = None) -> List[str]:
        """Generate permissions based on clearance level"""
        
        base_permissions = {
            SecurityLevel.PUBLIC: ['read_public_data'],
            SecurityLevel.INTERNAL: ['read_public_data', 'read_internal_data', 'basic_analysis'],
            SecurityLevel.CONFIDENTIAL: ['read_public_data', 'read_internal_data', 'basic_analysis', 
                                       'advanced_analysis', 'hypothesis_generation'],
            SecurityLevel.RESTRICTED: ['read_public_data', 'read_internal_data', 'basic_analysis',
                                     'advanced_analysis', 'hypothesis_generation', 'causal_discovery',
                                     'model_training'],
            SecurityLevel.TOP_SECRET: ['admin', 'all_operations']
        }
        
        granted_permissions = base_permissions.get(clearance, [])
        
        # Filter requested permissions
        if requested:
            granted_permissions = [p for p in granted_permissions if p in requested or 'admin' in granted_permissions]
        
        return granted_permissions
    
    def _check_rate_limit(self, user_id: str, operation: OperationType) -> bool:
        """Check if user is within rate limits"""
        
        rate_policy = self.security_policies["rate_limits"].get(operation)
        if not rate_policy:
            return True
        
        now = datetime.now()
        window_start = now - timedelta(minutes=rate_policy["window_minutes"])
        
        # Get recent requests for this user and operation
        user_key = f"{user_id}_{operation.value}"
        if user_key not in self.rate_limits:
            self.rate_limits[user_key] = []
        
        # Remove old requests
        self.rate_limits[user_key] = [req_time for req_time in self.rate_limits[user_key] 
                                     if req_time > window_start]
        
        # Check limit
        if len(self.rate_limits[user_key]) >= rate_policy["requests"]:
            return False
        
        # Add current request
        self.rate_limits[user_key].append(now)
        return True
    
    def _check_security_clearance(self, user_clearance: SecurityLevel, required: SecurityLevel) -> bool:
        """Check if user clearance meets requirement"""
        
        clearance_levels = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.RESTRICTED: 3,
            SecurityLevel.TOP_SECRET: 4
        }
        
        user_level = clearance_levels.get(user_clearance, 0)
        required_level = clearance_levels.get(required, 0)
        
        return user_level >= required_level
    
    def _get_operation_permission(self, operation: OperationType) -> str:
        """Get required permission for operation"""
        
        permission_mapping = {
            OperationType.DATA_ACCESS: "read_internal_data",
            OperationType.HYPOTHESIS_GENERATION: "hypothesis_generation",
            OperationType.EXPERIMENT_DESIGN: "basic_analysis",
            OperationType.RESULT_VALIDATION: "advanced_analysis",
            OperationType.MODEL_TRAINING: "model_training",
            OperationType.CAUSAL_DISCOVERY: "causal_discovery",
            OperationType.PUBLICATION_PREP: "advanced_analysis"
        }
        
        return permission_mapping.get(operation, "basic_analysis")
    
    def _check_resource_policies(self, credential: SecurityCredential, operation: OperationType, 
                               resource: str, security_level: SecurityLevel) -> bool:
        """Check resource-specific security policies"""
        
        # Example policies (in production, these would be more sophisticated)
        
        # Sensitive data access restrictions
        if "sensitive" in resource.lower() and credential.security_clearance.value in ["public", "internal"]:
            return False
        
        # Time-based restrictions
        current_hour = datetime.now().hour
        if security_level in [SecurityLevel.RESTRICTED, SecurityLevel.TOP_SECRET] and (current_hour < 6 or current_hour > 22):
            # Restricted operations only during business hours
            return False
        
        # Resource size limitations
        if operation == OperationType.MODEL_TRAINING and "large" in resource.lower():
            return credential.security_clearance in [SecurityLevel.RESTRICTED, SecurityLevel.TOP_SECRET]
        
        return True
    
    def _get_key_size_for_level(self, security_level: SecurityLevel) -> int:
        """Get encryption key size for security level"""
        return self.security_policies["encryption_requirements"].get(security_level, 256)
    
    def _store_encryption_key(self, key_id: str, key: bytes, security_level: SecurityLevel):
        """Store encryption key securely (simplified implementation)"""
        # In production, use a proper key management system
        key_storage_path = Path("secure_keys")
        key_storage_path.mkdir(exist_ok=True)
        
        with open(key_storage_path / f"{key_id}.key", "wb") as f:
            f.write(key)
    
    def _retrieve_encryption_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve encryption key"""
        key_storage_path = Path("secure_keys") / f"{key_id}.key"
        
        if key_storage_path.exists():
            with open(key_storage_path, "rb") as f:
                return f.read()
        
        return None
    
    def _log_security_event(self, user_id: str, operation: OperationType, result: str, 
                           security_level: SecurityLevel, details: Dict[str, Any] = None):
        """Log security event"""
        
        if not self.audit_enabled:
            return
        
        audit_log = SecurityAuditLog(
            timestamp=datetime.now(),
            user_id=user_id,
            operation=operation,
            resource=details.get("resource", "unknown") if details else "unknown",
            result=result,
            security_level=security_level,
            details=details or {}
        )
        
        self.security_audit_log.append(audit_log)
    
    def _detect_authentication_threats(self, user_id: str, credentials: Dict[str, Any]) -> List[ThreatDetection]:
        """Detect threats during authentication"""
        threats = []
        
        # Brute force detection (simplified)
        recent_failures = [log for log in self.security_audit_log
                          if log.user_id == user_id 
                          and log.result == "failure"
                          and log.timestamp > datetime.now() - timedelta(minutes=30)]
        
        if len(recent_failures) > 3:
            threats.append(ThreatDetection(
                threat_id=f"brute_force_{user_id}_{datetime.now().timestamp()}",
                threat_type="brute_force_attack",
                severity="high",
                description=f"Multiple failed authentication attempts for user {user_id}",
                indicators=[f"failed_attempts: {len(recent_failures)}"],
                recommended_actions=["lock_account", "require_2fa", "security_review"],
                confidence_score=0.9
            ))
        
        return threats
    
    def _detect_data_exfiltration(self, operation_data: Dict[str, Any]) -> Optional[ThreatDetection]:
        """Detect potential data exfiltration"""
        
        # Check for large data access patterns
        data_size = operation_data.get('data_size', 0)
        if data_size > 1000000:  # > 1MB
            return ThreatDetection(
                threat_id=f"data_exfil_{datetime.now().timestamp()}",
                threat_type="data_exfiltration",
                severity="medium",
                description="Large data access detected",
                indicators=[f"data_size: {data_size}"],
                recommended_actions=["review_access", "monitor_user"],
                confidence_score=0.6
            )
        
        return None
    
    def _detect_unusual_access_patterns(self, operation_data: Dict[str, Any], 
                                       context: Dict[str, Any] = None) -> Optional[ThreatDetection]:
        """Detect unusual access patterns"""
        
        # Time-based anomalies
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Outside business hours
            return ThreatDetection(
                threat_id=f"unusual_time_{datetime.now().timestamp()}",
                threat_type="unusual_access_time",
                severity="low",
                description="Access outside normal business hours",
                indicators=[f"access_hour: {current_hour}"],
                recommended_actions=["log_for_review", "verify_user"],
                confidence_score=0.4
            )
        
        return None
    
    def _detect_malicious_inputs(self, operation_data: Dict[str, Any]) -> Optional[ThreatDetection]:
        """Detect potentially malicious inputs"""
        
        # Check for injection patterns
        suspicious_patterns = ['<script>', 'eval(', 'exec(', '../', 'DROP TABLE']
        
        for key, value in operation_data.items():
            if isinstance(value, str):
                for pattern in suspicious_patterns:
                    if pattern in value.lower():
                        return ThreatDetection(
                            threat_id=f"malicious_input_{datetime.now().timestamp()}",
                            threat_type="malicious_input",
                            severity="high",
                            description="Potentially malicious input detected",
                            indicators=[f"pattern: {pattern}", f"field: {key}"],
                            recommended_actions=["block_request", "sanitize_input", "security_alert"],
                            confidence_score=0.8
                        )
        
        return None
    
    def _detect_privacy_violations(self, operation_data: Dict[str, Any]) -> Optional[ThreatDetection]:
        """Detect potential privacy violations"""
        
        # Check for PII patterns
        pii_patterns = ['ssn', 'social_security', 'credit_card', 'email', 'phone']
        
        for key, value in operation_data.items():
            if isinstance(value, str) and any(pattern in key.lower() for pattern in pii_patterns):
                return ThreatDetection(
                    threat_id=f"privacy_violation_{datetime.now().timestamp()}",
                    threat_type="privacy_violation",
                    severity="high",
                    description="Potential PII exposure detected",
                    indicators=[f"field: {key}"],
                    recommended_actions=["encrypt_data", "anonymize_data", "privacy_review"],
                    confidence_score=0.7
                )
        
        return None
    
    def _generate_security_recommendations(self, logs: List[SecurityAuditLog], 
                                         threats: List[ThreatDetection]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # High failure rate
        failure_rate = sum(1 for log in logs if log.result == "failure") / max(1, len(logs))
        if failure_rate > 0.1:
            recommendations.append("High failure rate detected - review authentication mechanisms")
        
        # Multiple high-severity threats
        high_severity_threats = [t for t in threats if t.severity == "high"]
        if len(high_severity_threats) > 5:
            recommendations.append("Multiple high-severity threats detected - conduct security review")
        
        # Encryption compliance
        encrypted_ops = sum(1 for log in logs if "encrypt" in log.details.get("operation", ""))
        if encrypted_ops / max(1, len(logs)) < 0.5:
            recommendations.append("Low encryption usage - enforce encryption policies")
        
        # Session management
        long_sessions = [log for log in logs if "session" in log.details.get("operation", "")]
        if len(long_sessions) > len(logs) * 0.3:
            recommendations.append("Many long-running sessions - implement session timeouts")
        
        return recommendations