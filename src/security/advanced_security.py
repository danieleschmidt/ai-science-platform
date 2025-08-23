"""Advanced security module for research platform"""

import hashlib
import secrets
import hmac
import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import re
from functools import wraps

from ..utils.error_handling import robust_execution, SecurityError
from ..utils.validation import ValidationMixin

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_type: str
    user_id: Optional[str]
    resource: str
    action: str
    outcome: str  # 'success', 'failure', 'blocked'
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: int = 0  # 0-100


@dataclass
class AccessToken:
    """Secure access token"""
    token_id: str
    user_id: str
    scopes: List[str]
    issued_at: datetime
    expires_at: datetime
    ip_restrictions: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecureKeyManager:
    """Secure key management and encryption"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        """Initialize secure key manager"""
        
        if master_key:
            self.master_key = master_key
        else:
            # Generate new master key
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
        
        # Key derivation for different purposes
        self.derived_keys = {}
        
        logger.info("Secure key manager initialized")
    
    def derive_key(self, purpose: str, salt: Optional[bytes] = None) -> bytes:
        """Derive key for specific purpose"""
        
        if purpose in self.derived_keys:
            return self.derived_keys[purpose]
        
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(self.master_key)
        self.derived_keys[purpose] = key
        
        return key
    
    def encrypt_sensitive_data(self, data: Union[str, bytes], context: str = "") -> str:
        """Encrypt sensitive data with context"""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Add context for additional security
        if context:
            context_hash = hashlib.sha256(context.encode()).digest()[:8]
            data = context_hash + data
        
        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt_sensitive_data(self, encrypted_data: str, context: str = "") -> bytes:
        """Decrypt sensitive data with context verification"""
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.fernet.decrypt(encrypted_bytes)
            
            if context:
                context_hash = hashlib.sha256(context.encode()).digest()[:8]
                if not decrypted.startswith(context_hash):
                    raise SecurityError("Context verification failed")
                decrypted = decrypted[8:]
            
            return decrypted
            
        except Exception as e:
            raise SecurityError(f"Decryption failed: {str(e)}")
    
    def create_secure_hash(self, data: Union[str, bytes], salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Create secure hash with salt"""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 for password-like data
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        hash_value = kdf.derive(data)
        
        return (
            base64.urlsafe_b64encode(hash_value).decode('utf-8'),
            base64.urlsafe_b64encode(salt).decode('utf-8')
        )
    
    def verify_secure_hash(self, data: Union[str, bytes], hash_value: str, salt: str) -> bool:
        """Verify secure hash"""
        
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            salt_bytes = base64.urlsafe_b64decode(salt.encode('utf-8'))
            expected_hash = base64.urlsafe_b64decode(hash_value.encode('utf-8'))
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
            )
            
            computed_hash = kdf.derive(data)
            return hmac.compare_digest(expected_hash, computed_hash)
            
        except Exception:
            return False


class RateLimiter:
    """Advanced rate limiting with different strategies"""
    
    def __init__(self):
        """Initialize rate limiter"""
        self.token_buckets = {}
        self.fixed_windows = {}
        self.sliding_windows = {}
        
    def check_rate_limit(self, 
                        identifier: str, 
                        strategy: str = "token_bucket",
                        limit: int = 100,
                        window_seconds: int = 3600,
                        burst_size: Optional[int] = None) -> bool:
        """Check if action is within rate limits"""
        
        current_time = time.time()
        
        if strategy == "token_bucket":
            return self._check_token_bucket(identifier, limit, window_seconds, burst_size or limit, current_time)
        elif strategy == "fixed_window":
            return self._check_fixed_window(identifier, limit, window_seconds, current_time)
        elif strategy == "sliding_window":
            return self._check_sliding_window(identifier, limit, window_seconds, current_time)
        else:
            logger.warning(f"Unknown rate limiting strategy: {strategy}")
            return True
    
    def _check_token_bucket(self, identifier: str, rate: int, window: int, capacity: int, current_time: float) -> bool:
        """Token bucket rate limiting"""
        
        if identifier not in self.token_buckets:
            self.token_buckets[identifier] = {
                'tokens': capacity,
                'last_refill': current_time
            }
        
        bucket = self.token_buckets[identifier]
        
        # Refill tokens
        time_passed = current_time - bucket['last_refill']
        tokens_to_add = (time_passed / window) * rate
        bucket['tokens'] = min(capacity, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = current_time
        
        # Check if we can consume a token
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        
        return False
    
    def _check_fixed_window(self, identifier: str, limit: int, window: int, current_time: float) -> bool:
        """Fixed window rate limiting"""
        
        window_start = int(current_time // window) * window
        
        if identifier not in self.fixed_windows:
            self.fixed_windows[identifier] = {}
        
        if window_start not in self.fixed_windows[identifier]:
            # Clean old windows
            old_windows = [w for w in self.fixed_windows[identifier].keys() if w < window_start - window]
            for old_window in old_windows:
                del self.fixed_windows[identifier][old_window]
            
            self.fixed_windows[identifier][window_start] = 0
        
        if self.fixed_windows[identifier][window_start] < limit:
            self.fixed_windows[identifier][window_start] += 1
            return True
        
        return False
    
    def _check_sliding_window(self, identifier: str, limit: int, window: int, current_time: float) -> bool:
        """Sliding window rate limiting"""
        
        if identifier not in self.sliding_windows:
            self.sliding_windows[identifier] = []
        
        # Remove old entries
        cutoff_time = current_time - window
        self.sliding_windows[identifier] = [
            timestamp for timestamp in self.sliding_windows[identifier]
            if timestamp > cutoff_time
        ]
        
        if len(self.sliding_windows[identifier]) < limit:
            self.sliding_windows[identifier].append(current_time)
            return True
        
        return False


class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self):
        """Initialize input validator"""
        
        # Common patterns
        self.patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
            'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
            'safe_string': re.compile(r'^[a-zA-Z0-9_\-\s\.]+$'),
            'sql_injection': re.compile(r'(union|select|insert|update|delete|drop|exec|script)', re.IGNORECASE),
            'xss': re.compile(r'<script|javascript:|on\w+\s*=', re.IGNORECASE)
        }
        
        # Dangerous patterns to block
        self.dangerous_patterns = [
            'sql_injection',
            'xss'
        ]
    
    def validate_and_sanitize(self, 
                            value: Any, 
                            validation_type: str,
                            max_length: Optional[int] = None,
                            allow_empty: bool = False) -> Any:
        """Validate and sanitize input value"""
        
        if value is None:
            if allow_empty:
                return None
            else:
                raise SecurityError("Value cannot be None")
        
        # Convert to string for validation
        str_value = str(value)
        
        # Length check
        if max_length and len(str_value) > max_length:
            raise SecurityError(f"Value exceeds maximum length of {max_length}")
        
        # Empty check
        if not allow_empty and not str_value.strip():
            raise SecurityError("Value cannot be empty")
        
        # Check for dangerous patterns
        for pattern_name in self.dangerous_patterns:
            if pattern_name in self.patterns and self.patterns[pattern_name].search(str_value):
                raise SecurityError(f"Input contains dangerous pattern: {pattern_name}")
        
        # Validate against specific pattern
        if validation_type in self.patterns:
            if not self.patterns[validation_type].match(str_value):
                raise SecurityError(f"Value does not match required pattern: {validation_type}")
        
        # Type-specific sanitization
        if validation_type == 'safe_string':
            return self._sanitize_string(str_value)
        elif validation_type == 'alphanumeric':
            return ''.join(c for c in str_value if c.isalnum())
        elif validation_type in ['int', 'integer']:
            try:
                return int(str_value)
            except ValueError:
                raise SecurityError("Value is not a valid integer")
        elif validation_type in ['float', 'number']:
            try:
                return float(str_value)
            except ValueError:
                raise SecurityError("Value is not a valid number")
        
        return str_value
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input"""
        
        # Remove control characters
        sanitized = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')
        
        # Limit consecutive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Trim
        sanitized = sanitized.strip()
        
        return sanitized
    
    def validate_file_upload(self, 
                           filename: str, 
                           content: bytes,
                           allowed_extensions: List[str],
                           max_size_mb: int = 10) -> Dict[str, Any]:
        """Validate file upload"""
        
        result = {
            'valid': True,
            'filename': filename,
            'size_mb': len(content) / (1024 * 1024),
            'errors': []
        }
        
        # Filename validation
        if not filename or '..' in filename:
            result['valid'] = False
            result['errors'].append("Invalid filename")
        
        # Extension validation
        file_ext = Path(filename).suffix.lower()
        if file_ext not in [f".{ext.lower()}" for ext in allowed_extensions]:
            result['valid'] = False
            result['errors'].append(f"File type not allowed. Allowed: {allowed_extensions}")
        
        # Size validation
        if result['size_mb'] > max_size_mb:
            result['valid'] = False
            result['errors'].append(f"File too large. Maximum: {max_size_mb}MB")
        
        # Content validation (basic magic bytes check)
        if not self._validate_file_content(content, file_ext):
            result['valid'] = False
            result['errors'].append("File content does not match extension")
        
        return result
    
    def _validate_file_content(self, content: bytes, expected_ext: str) -> bool:
        """Validate file content against extension using magic bytes"""
        
        if len(content) < 4:
            return False
        
        magic_bytes = {
            '.pdf': [b'%PDF'],
            '.png': [b'\x89PNG'],
            '.jpg': [b'\xff\xd8\xff'],
            '.jpeg': [b'\xff\xd8\xff'],
            '.gif': [b'GIF87a', b'GIF89a'],
            '.txt': [],  # Text files don't have reliable magic bytes
            '.json': [],  # JSON files don't have magic bytes
            '.csv': []   # CSV files don't have magic bytes
        }
        
        if expected_ext not in magic_bytes:
            return True  # Allow unknown extensions for now
        
        expected_magics = magic_bytes[expected_ext]
        if not expected_magics:
            return True  # No magic bytes to check
        
        return any(content.startswith(magic) for magic in expected_magics)


class AuditLogger:
    """Security audit logging system"""
    
    def __init__(self, log_file: Optional[Path] = None):
        """Initialize audit logger"""
        
        self.log_file = log_file or Path("security_audit.log")
        self.events = []
        
        # Ensure log file directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log_security_event(self, event: SecurityEvent):
        """Log security event"""
        
        self.events.append(event)
        
        # Write to file
        try:
            with open(self.log_file, 'a') as f:
                event_data = {
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'user_id': event.user_id,
                    'resource': event.resource,
                    'action': event.action,
                    'outcome': event.outcome,
                    'ip_address': event.ip_address,
                    'risk_score': event.risk_score,
                    'details': event.details
                }
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def log_authentication_attempt(self, user_id: str, success: bool, ip_address: str = None, details: Dict[str, Any] = None):
        """Log authentication attempt"""
        
        event = SecurityEvent(
            event_type='authentication',
            user_id=user_id,
            resource='auth_system',
            action='login',
            outcome='success' if success else 'failure',
            ip_address=ip_address,
            details=details or {},
            risk_score=0 if success else 30
        )
        
        self.log_security_event(event)
    
    def log_authorization_check(self, user_id: str, resource: str, action: str, granted: bool, details: Dict[str, Any] = None):
        """Log authorization check"""
        
        event = SecurityEvent(
            event_type='authorization',
            user_id=user_id,
            resource=resource,
            action=action,
            outcome='success' if granted else 'blocked',
            details=details or {},
            risk_score=0 if granted else 20
        )
        
        self.log_security_event(event)
    
    def log_data_access(self, user_id: str, resource: str, action: str, success: bool, details: Dict[str, Any] = None):
        """Log data access attempt"""
        
        event = SecurityEvent(
            event_type='data_access',
            user_id=user_id,
            resource=resource,
            action=action,
            outcome='success' if success else 'failure',
            details=details or {},
            risk_score=10 if not success else 0
        )
        
        self.log_security_event(event)
    
    def get_security_events(self, 
                          hours: int = 24,
                          event_type: Optional[str] = None,
                          user_id: Optional[str] = None) -> List[SecurityEvent]:
        """Get security events from specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_events = [
            event for event in self.events
            if event.timestamp >= cutoff_time
        ]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        return filtered_events
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security events summary"""
        
        events = self.get_security_events(hours)
        
        if not events:
            return {"total_events": 0, "time_period_hours": hours}
        
        # Aggregate statistics
        event_types = {}
        outcomes = {}
        risk_scores = []
        users = set()
        
        for event in events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            outcomes[event.outcome] = outcomes.get(event.outcome, 0) + 1
            risk_scores.append(event.risk_score)
            if event.user_id:
                users.add(event.user_id)
        
        return {
            "total_events": len(events),
            "time_period_hours": hours,
            "event_types": event_types,
            "outcomes": outcomes,
            "unique_users": len(users),
            "avg_risk_score": sum(risk_scores) / len(risk_scores) if risk_scores else 0,
            "max_risk_score": max(risk_scores) if risk_scores else 0,
            "high_risk_events": sum(1 for score in risk_scores if score >= 50)
        }


class SecureTokenManager:
    """Secure token management for API access"""
    
    def __init__(self, key_manager: SecureKeyManager):
        """Initialize token manager"""
        
        self.key_manager = key_manager
        self.active_tokens = {}
        self.revoked_tokens = set()
        
    def create_access_token(self, 
                          user_id: str,
                          scopes: List[str],
                          duration_hours: int = 24,
                          ip_restrictions: Optional[List[str]] = None) -> AccessToken:
        """Create secure access token"""
        
        token_id = secrets.token_urlsafe(32)
        now = datetime.now()
        
        token = AccessToken(
            token_id=token_id,
            user_id=user_id,
            scopes=scopes,
            issued_at=now,
            expires_at=now + timedelta(hours=duration_hours),
            ip_restrictions=ip_restrictions,
            metadata={'created_by': 'system'}
        )
        
        self.active_tokens[token_id] = token
        
        logger.info(f"Created access token for user {user_id} with scopes {scopes}")
        return token
    
    def validate_token(self, token_id: str, required_scope: Optional[str] = None, client_ip: Optional[str] = None) -> Optional[AccessToken]:
        """Validate access token"""
        
        if token_id in self.revoked_tokens:
            return None
        
        if token_id not in self.active_tokens:
            return None
        
        token = self.active_tokens[token_id]
        
        # Check expiration
        if datetime.now() > token.expires_at:
            del self.active_tokens[token_id]
            return None
        
        # Check scope
        if required_scope and required_scope not in token.scopes:
            return None
        
        # Check IP restrictions
        if token.ip_restrictions and client_ip:
            if client_ip not in token.ip_restrictions:
                return None
        
        return token
    
    def revoke_token(self, token_id: str):
        """Revoke access token"""
        
        self.revoked_tokens.add(token_id)
        if token_id in self.active_tokens:
            del self.active_tokens[token_id]
        
        logger.info(f"Revoked access token {token_id}")
    
    def revoke_user_tokens(self, user_id: str):
        """Revoke all tokens for a user"""
        
        tokens_to_revoke = [
            token_id for token_id, token in self.active_tokens.items()
            if token.user_id == user_id
        ]
        
        for token_id in tokens_to_revoke:
            self.revoke_token(token_id)
        
        logger.info(f"Revoked {len(tokens_to_revoke)} tokens for user {user_id}")
    
    def cleanup_expired_tokens(self):
        """Remove expired tokens"""
        
        now = datetime.now()
        expired_tokens = [
            token_id for token_id, token in self.active_tokens.items()
            if now > token.expires_at
        ]
        
        for token_id in expired_tokens:
            del self.active_tokens[token_id]
        
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")


class SecurityManager(ValidationMixin):
    """Central security management system"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        """Initialize security manager"""
        
        self.key_manager = SecureKeyManager(master_key)
        self.rate_limiter = RateLimiter()
        self.input_validator = InputValidator()
        self.audit_logger = AuditLogger()
        self.token_manager = SecureTokenManager(self.key_manager)
        
        # Security policies
        self.policies = {
            'max_login_attempts': 5,
            'lockout_duration_minutes': 30,
            'password_min_length': 8,
            'token_expiry_hours': 24,
            'rate_limit_per_hour': 1000,
            'require_https': True
        }
        
        # Failed login tracking
        self.failed_logins = {}
        self.locked_accounts = {}
        
        logger.info("Security manager initialized")
    
    @robust_execution(recovery_strategy='fail_secure')
    def authenticate_user(self, user_id: str, password: str, ip_address: str = None) -> Dict[str, Any]:
        """Authenticate user with comprehensive security checks"""
        
        # Check if account is locked
        if self._is_account_locked(user_id):
            self.audit_logger.log_authentication_attempt(
                user_id, False, ip_address, 
                {'reason': 'account_locked'}
            )
            return {'success': False, 'reason': 'account_locked'}
        
        # Rate limiting
        if not self.rate_limiter.check_rate_limit(
            f"login:{user_id}", "token_bucket", 5, 3600
        ):
            self.audit_logger.log_authentication_attempt(
                user_id, False, ip_address,
                {'reason': 'rate_limited'}
            )
            return {'success': False, 'reason': 'rate_limited'}
        
        # Validate inputs
        try:
            user_id = self.input_validator.validate_and_sanitize(user_id, 'safe_string', max_length=100)
            password = self.input_validator.validate_and_sanitize(password, 'safe_string', max_length=200)
        except SecurityError as e:
            self.audit_logger.log_authentication_attempt(
                user_id, False, ip_address,
                {'reason': 'invalid_input', 'error': str(e)}
            )
            return {'success': False, 'reason': 'invalid_input'}
        
        # Simulate user authentication (in real implementation, check against database)
        # For demo purposes, accept any user with password length >= 8
        auth_success = len(password) >= self.policies['password_min_length']
        
        if auth_success:
            # Reset failed login counter
            if user_id in self.failed_logins:
                del self.failed_logins[user_id]
            
            # Create access token
            token = self.token_manager.create_access_token(
                user_id=user_id,
                scopes=['read', 'write'],
                duration_hours=self.policies['token_expiry_hours']
            )
            
            self.audit_logger.log_authentication_attempt(
                user_id, True, ip_address,
                {'token_id': token.token_id}
            )
            
            return {
                'success': True,
                'token': token.token_id,
                'expires_at': token.expires_at.isoformat(),
                'scopes': token.scopes
            }
        else:
            # Track failed login
            self._track_failed_login(user_id)
            
            self.audit_logger.log_authentication_attempt(
                user_id, False, ip_address,
                {'reason': 'invalid_credentials'}
            )
            
            return {'success': False, 'reason': 'invalid_credentials'}
    
    def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked due to failed login attempts"""
        
        if user_id not in self.locked_accounts:
            return False
        
        lock_expires = self.locked_accounts[user_id]
        if datetime.now() > lock_expires:
            del self.locked_accounts[user_id]
            return False
        
        return True
    
    def _track_failed_login(self, user_id: str):
        """Track failed login attempts and lock account if necessary"""
        
        if user_id not in self.failed_logins:
            self.failed_logins[user_id] = []
        
        self.failed_logins[user_id].append(datetime.now())
        
        # Remove attempts older than 1 hour
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.failed_logins[user_id] = [
            attempt for attempt in self.failed_logins[user_id]
            if attempt > cutoff_time
        ]
        
        # Lock account if too many failures
        if len(self.failed_logins[user_id]) >= self.policies['max_login_attempts']:
            lock_duration = timedelta(minutes=self.policies['lockout_duration_minutes'])
            self.locked_accounts[user_id] = datetime.now() + lock_duration
            logger.warning(f"Account locked due to failed login attempts: {user_id}")
    
    @robust_execution(recovery_strategy='fail_secure')
    def authorize_action(self, token_id: str, resource: str, action: str, client_ip: str = None) -> Dict[str, Any]:
        """Authorize action with token-based security"""
        
        # Validate token
        token = self.token_manager.validate_token(token_id, client_ip=client_ip)
        
        if not token:
            self.audit_logger.log_authorization_check(
                None, resource, action, False,
                {'reason': 'invalid_token'}
            )
            return {'authorized': False, 'reason': 'invalid_token'}
        
        # Check if action is allowed for user's scopes
        required_scope = self._determine_required_scope(resource, action)
        
        if required_scope and required_scope not in token.scopes:
            self.audit_logger.log_authorization_check(
                token.user_id, resource, action, False,
                {'reason': 'insufficient_scope', 'required': required_scope, 'available': token.scopes}
            )
            return {'authorized': False, 'reason': 'insufficient_scope'}
        
        # Rate limiting per user
        user_rate_key = f"user_actions:{token.user_id}"
        if not self.rate_limiter.check_rate_limit(
            user_rate_key, "sliding_window", 
            self.policies['rate_limit_per_hour'], 3600
        ):
            self.audit_logger.log_authorization_check(
                token.user_id, resource, action, False,
                {'reason': 'rate_limited'}
            )
            return {'authorized': False, 'reason': 'rate_limited'}
        
        # Authorization successful
        self.audit_logger.log_authorization_check(
            token.user_id, resource, action, True,
            {'token_id': token_id}
        )
        
        return {
            'authorized': True,
            'user_id': token.user_id,
            'scopes': token.scopes
        }
    
    def _determine_required_scope(self, resource: str, action: str) -> Optional[str]:
        """Determine required scope for resource and action"""
        
        # Simple scope mapping
        scope_mapping = {
            'read': ['get', 'list', 'view'],
            'write': ['create', 'update', 'modify'],
            'admin': ['delete', 'admin', 'manage']
        }
        
        action_lower = action.lower()
        
        for scope, actions in scope_mapping.items():
            if any(a in action_lower for a in actions):
                return scope
        
        return 'read'  # Default to read scope
    
    def secure_data_access(self, user_id: str, resource: str, action: str, data: Any) -> Dict[str, Any]:
        """Secure data access with encryption and audit logging"""
        
        try:
            # Log data access attempt
            self.audit_logger.log_data_access(user_id, resource, action, True, {
                'data_type': type(data).__name__,
                'data_size': len(str(data))
            })
            
            # Encrypt sensitive data if needed
            if self._is_sensitive_resource(resource):
                if isinstance(data, (str, bytes)):
                    encrypted_data = self.key_manager.encrypt_sensitive_data(data, context=resource)
                    return {
                        'success': True,
                        'data': encrypted_data,
                        'encrypted': True
                    }
            
            return {
                'success': True,
                'data': data,
                'encrypted': False
            }
            
        except Exception as e:
            self.audit_logger.log_data_access(user_id, resource, action, False, {
                'error': str(e)
            })
            return {'success': False, 'error': str(e)}
    
    def _is_sensitive_resource(self, resource: str) -> bool:
        """Check if resource contains sensitive data"""
        
        sensitive_patterns = [
            'password', 'token', 'key', 'secret', 
            'credential', 'private', 'confidential'
        ]
        
        resource_lower = resource.lower()
        return any(pattern in resource_lower for pattern in sensitive_patterns)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        
        # Token statistics
        active_tokens = len(self.token_manager.active_tokens)
        revoked_tokens = len(self.token_manager.revoked_tokens)
        
        # Account security
        locked_accounts = len(self.locked_accounts)
        accounts_with_failed_logins = len(self.failed_logins)
        
        # Audit summary
        audit_summary = self.audit_logger.get_security_summary(24)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'tokens': {
                'active': active_tokens,
                'revoked': revoked_tokens
            },
            'accounts': {
                'locked': locked_accounts,
                'with_failed_logins': accounts_with_failed_logins
            },
            'audit_summary': audit_summary,
            'security_policies': self.policies
        }
    
    def run_security_maintenance(self):
        """Run periodic security maintenance tasks"""
        
        logger.info("Running security maintenance")
        
        # Cleanup expired tokens
        self.token_manager.cleanup_expired_tokens()
        
        # Unlock expired account locks
        current_time = datetime.now()
        expired_locks = [
            user_id for user_id, unlock_time in self.locked_accounts.items()
            if current_time > unlock_time
        ]
        
        for user_id in expired_locks:
            del self.locked_accounts[user_id]
            logger.info(f"Unlocked account: {user_id}")
        
        # Clean old failed login records
        cutoff_time = current_time - timedelta(hours=24)
        for user_id in list(self.failed_logins.keys()):
            self.failed_logins[user_id] = [
                attempt for attempt in self.failed_logins[user_id]
                if attempt > cutoff_time
            ]
            if not self.failed_logins[user_id]:
                del self.failed_logins[user_id]
        
        logger.info("Security maintenance completed")


def require_authentication(security_manager: SecurityManager):
    """Decorator for requiring authentication"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract token from kwargs or headers
            token_id = kwargs.get('token_id') or kwargs.get('authorization')
            
            if not token_id:
                raise SecurityError("Authentication token required")
            
            # Clean token format (remove Bearer prefix if present)
            if token_id.startswith('Bearer '):
                token_id = token_id[7:]
            
            # Validate token
            token = security_manager.token_manager.validate_token(token_id)
            if not token:
                raise SecurityError("Invalid or expired authentication token")
            
            # Add user context to kwargs
            kwargs['current_user'] = token.user_id
            kwargs['user_scopes'] = token.scopes
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_authorization(security_manager: SecurityManager, resource: str, action: str):
    """Decorator for requiring authorization"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            token_id = kwargs.get('token_id') or kwargs.get('authorization')
            client_ip = kwargs.get('client_ip')
            
            if token_id and token_id.startswith('Bearer '):
                token_id = token_id[7:]
            
            auth_result = security_manager.authorize_action(
                token_id, resource, action, client_ip
            )
            
            if not auth_result['authorized']:
                raise SecurityError(f"Authorization failed: {auth_result['reason']}")
            
            # Add authorization context
            kwargs['current_user'] = auth_result['user_id']
            kwargs['user_scopes'] = auth_result['scopes']
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator