"""Secure configuration management example"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class SecureConfig:
    """Secure configuration using environment variables"""
    
    # API Keys - load from environment
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN", "")
    
    # Database credentials
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "ai_science")
    db_user: str = os.getenv("DB_USER", "")
    db_password: str = os.getenv("DB_PASSWORD", "")
    
    # Security settings
    secret_key: str = os.getenv("SECRET_KEY", "")
    jwt_secret: str = os.getenv("JWT_SECRET", "")
    
    def validate(self) -> None:
        """Validate that required secrets are present"""
        required_secrets = [
            ("SECRET_KEY", self.secret_key),
            ("JWT_SECRET", self.jwt_secret),
        ]
        
        missing = [name for name, value in required_secrets if not value]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
    
    @classmethod
    def load(cls) -> 'SecureConfig':
        """Load configuration and validate"""
        config = cls()
        config.validate()
        return config

# Example .env file (DO NOT COMMIT TO VERSION CONTROL):
# SECRET_KEY=your-super-secret-key-here
# JWT_SECRET=your-jwt-secret-here  
# OPENAI_API_KEY=your-api-key
# DB_PASSWORD=your-db-password
