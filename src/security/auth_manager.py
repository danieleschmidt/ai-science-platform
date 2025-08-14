"""Authentication and authorization management"""

import hashlib
import secrets
import time
import jwt
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class Permission(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    DISCOVER = "discover"
    EXPERIMENT = "experiment"
    API_ACCESS = "api_access"


@dataclass
class Role:
    """User role with permissions"""
    name: str
    permissions: Set[Permission]
    description: str = ""
    is_active: bool = True


@dataclass
class User:
    """User account information"""
    username: str
    email: str
    password_hash: str
    roles: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    failed_login_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """User session information"""
    session_id: str
    username: str
    created_at: float
    last_activity: float
    ip_address: str = ""
    user_agent: str = ""
    permissions: Set[Permission] = field(default_factory=set)


class AuthManager:
    """Comprehensive authentication and authorization manager"""
    
    def __init__(self, secret_key: str = None, session_timeout: float = 3600):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.session_timeout = session_timeout
        
        # Storage
        self.users = {}
        self.roles = {}
        self.sessions = {}
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        self.password_min_length = 8
        self.require_special_chars = True
        
        # Setup default roles
        self._create_default_roles()
        
        logger.info("AuthManager initialized")
    
    def _create_default_roles(self) -> None:
        """Create default user roles"""
        default_roles = [
            Role(
                name="admin",
                permissions={Permission.READ, Permission.WRITE, Permission.EXECUTE, 
                           Permission.ADMIN, Permission.DISCOVER, Permission.EXPERIMENT, 
                           Permission.API_ACCESS},
                description="Full system administrator"
            ),
            Role(
                name="researcher",
                permissions={Permission.READ, Permission.DISCOVER, Permission.EXPERIMENT, 
                           Permission.API_ACCESS},
                description="Research scientist with discovery capabilities"
            ),
            Role(
                name="analyst",
                permissions={Permission.READ, Permission.API_ACCESS},
                description="Data analyst with read-only access"
            ),
            Role(
                name="api_user",
                permissions={Permission.API_ACCESS, Permission.READ},
                description="API-only access for external systems"
            )
        ]
        
        for role in default_roles:
            self.roles[role.name] = role
    
    def create_user(self, username: str, email: str, password: str, 
                   roles: List[str] = None) -> bool:
        """Create a new user account"""
        
        # Validate input
        if not self._validate_username(username):
            logger.warning(f"Invalid username: {username}")
            return False
        
        if not self._validate_email(email):
            logger.warning(f"Invalid email: {email}")
            return False
        
        if not self._validate_password(password):
            logger.warning(f"Invalid password for user: {username}")
            return False
        
        if username in self.users:
            logger.warning(f"User already exists: {username}")
            return False
        
        # Create user
        password_hash = self._hash_password(password)
        user_roles = roles or ["researcher"]
        
        # Validate roles
        for role in user_roles:
            if role not in self.roles:
                logger.warning(f"Invalid role: {role}")
                return False
        
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            roles=user_roles
        )
        
        self.users[username] = user
        logger.info(f"User created: {username}")
        return True
    
    def authenticate(self, username: str, password: str, 
                    ip_address: str = "", user_agent: str = "") -> Optional[str]:
        """Authenticate user and create session"""
        
        if username not in self.users:
            logger.warning(f"Authentication failed: user not found: {username}")
            return None
        
        user = self.users[username]
        
        # Check if account is locked
        if not user.is_active:
            logger.warning(f"Authentication failed: account inactive: {username}")
            return None
        
        if user.failed_login_attempts >= self.max_failed_attempts:
            logger.warning(f"Authentication failed: account locked: {username}")
            return None
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            logger.warning(f"Authentication failed: invalid password: {username}")
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.last_login = time.time()
        
        # Create session
        session_id = self._generate_session_id()
        permissions = self._get_user_permissions(user)
        
        session = Session(
            session_id=session_id,
            username=username,
            created_at=time.time(),
            last_activity=time.time(),
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=permissions
        )
        
        self.sessions[session_id] = session
        logger.info(f"User authenticated: {username}")
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate and refresh session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        current_time = time.time()
        
        # Check timeout
        if current_time - session.last_activity > self.session_timeout:
            self.logout(session_id)
            return None
        
        # Update activity
        session.last_activity = current_time
        return session
    
    def logout(self, session_id: str) -> bool:
        """Logout user and destroy session"""
        if session_id in self.sessions:
            username = self.sessions[session_id].username
            del self.sessions[session_id]
            logger.info(f"User logged out: {username}")
            return True
        return False
    
    def check_permission(self, session_id: str, permission: Permission) -> bool:
        """Check if session has specific permission"""
        session = self.validate_session(session_id)
        if not session:
            return False
        
        return permission in session.permissions
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract session_id from kwargs or request context
                session_id = kwargs.get('session_id')
                if not session_id:
                    raise PermissionError("No session ID provided")
                
                if not self.check_permission(session_id, permission):
                    raise PermissionError(f"Permission required: {permission.value}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_auth(self):
        """Decorator to require authentication"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                session_id = kwargs.get('session_id')
                if not session_id:
                    raise PermissionError("Authentication required")
                
                session = self.validate_session(session_id)
                if not session:
                    raise PermissionError("Invalid or expired session")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def create_role(self, name: str, permissions: Set[Permission], 
                   description: str = "") -> bool:
        """Create a new role"""
        if name in self.roles:
            logger.warning(f"Role already exists: {name}")
            return False
        
        role = Role(
            name=name,
            permissions=permissions,
            description=description
        )
        
        self.roles[name] = role
        logger.info(f"Role created: {name}")
        return True
    
    def assign_role(self, username: str, role_name: str) -> bool:
        """Assign role to user"""
        if username not in self.users:
            logger.warning(f"User not found: {username}")
            return False
        
        if role_name not in self.roles:
            logger.warning(f"Role not found: {role_name}")
            return False
        
        user = self.users[username]
        if role_name not in user.roles:
            user.roles.append(role_name)
            logger.info(f"Role {role_name} assigned to user {username}")
        
        return True
    
    def revoke_role(self, username: str, role_name: str) -> bool:
        """Revoke role from user"""
        if username not in self.users:
            return False
        
        user = self.users[username]
        if role_name in user.roles:
            user.roles.remove(role_name)
            logger.info(f"Role {role_name} revoked from user {username}")
        
        return True
    
    def change_password(self, username: str, old_password: str, 
                       new_password: str) -> bool:
        """Change user password"""
        if username not in self.users:
            return False
        
        user = self.users[username]
        
        # Verify old password
        if not self._verify_password(old_password, user.password_hash):
            logger.warning(f"Password change failed: invalid old password: {username}")
            return False
        
        # Validate new password
        if not self._validate_password(new_password):
            logger.warning(f"Password change failed: invalid new password: {username}")
            return False
        
        # Update password
        user.password_hash = self._hash_password(new_password)
        logger.info(f"Password changed for user: {username}")
        return True
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate user account"""
        if username not in self.users:
            return False
        
        user = self.users[username]
        user.is_active = False
        
        # Logout all sessions for this user
        sessions_to_remove = [sid for sid, session in self.sessions.items() 
                            if session.username == username]
        for sid in sessions_to_remove:
            del self.sessions[sid]
        
        logger.info(f"User deactivated: {username}")
        return True
    
    def activate_user(self, username: str) -> bool:
        """Activate user account"""
        if username not in self.users:
            return False
        
        user = self.users[username]
        user.is_active = True
        user.failed_login_attempts = 0
        
        logger.info(f"User activated: {username}")
        return True
    
    def generate_api_token(self, username: str, expires_in: float = 86400) -> Optional[str]:
        """Generate API token for user"""
        if username not in self.users:
            return None
        
        user = self.users[username]
        permissions = self._get_user_permissions(user)
        
        payload = {
            "username": username,
            "permissions": [p.value for p in permissions],
            "iat": time.time(),
            "exp": time.time() + expires_in
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        logger.info(f"API token generated for user: {username}")
        return token
    
    def validate_api_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate API token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            username = payload.get("username")
            if username not in self.users:
                return None
            
            user = self.users[username]
            if not user.is_active:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("API token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid API token")
            return None
    
    def _validate_username(self, username: str) -> bool:
        """Validate username"""
        return (len(username) >= 3 and 
                len(username) <= 50 and
                username.isalnum())
    
    def _validate_email(self, email: str) -> bool:
        """Validate email address"""
        return "@" in email and "." in email.split("@")[1]
    
    def _validate_password(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < self.password_min_length:
            return False
        
        if self.require_special_chars:
            has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
            has_digit = any(c.isdigit() for c in password)
            has_upper = any(c.isupper() for c in password)
            has_lower = any(c.islower() for c in password)
            
            return has_special and has_digit and has_upper and has_lower
        
        return True
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{pwd_hash.hex()}"
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        try:
            salt, pwd_hash = stored_hash.split(':')
            new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return new_hash.hex() == pwd_hash
        except ValueError:
            return False
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)
    
    def _get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all permissions for user based on roles"""
        permissions = set()
        
        for role_name in user.roles:
            if role_name in self.roles:
                role = self.roles[role_name]
                if role.is_active:
                    permissions.update(role.permissions)
        
        return permissions
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information (excluding sensitive data)"""
        if username not in self.users:
            return None
        
        user = self.users[username]
        return {
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "is_active": user.is_active,
            "created_at": user.created_at,
            "last_login": user.last_login,
            "metadata": user.metadata
        }
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users (excluding sensitive data)"""
        return [self.get_user_info(username) for username in self.users.keys()]
    
    def list_roles(self) -> List[Dict[str, Any]]:
        """List all roles"""
        return [
            {
                "name": role.name,
                "permissions": [p.value for p in role.permissions],
                "description": role.description,
                "is_active": role.is_active
            }
            for role in self.roles.values()
        ]
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get information about active sessions"""
        return [
            {
                "session_id": session.session_id[:8] + "...",  # Truncated for security
                "username": session.username,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "ip_address": session.ip_address,
                "user_agent": session.user_agent[:50] + "..." if len(session.user_agent) > 50 else session.user_agent
            }
            for session in self.sessions.values()
        ]
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time - session.last_activity > self.session_timeout
        ]
        
        for sid in expired_sessions:
            del self.sessions[sid]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        return {
            "total_users": len(self.users),
            "active_users": sum(1 for user in self.users.values() if user.is_active),
            "total_roles": len(self.roles),
            "active_sessions": len(self.sessions),
            "failed_login_attempts": sum(user.failed_login_attempts for user in self.users.values()),
            "locked_accounts": sum(1 for user in self.users.values() 
                                 if user.failed_login_attempts >= self.max_failed_attempts)
        }