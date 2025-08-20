#!/usr/bin/env python3
"""Fix remaining security issues"""

import os
import re
from pathlib import Path

def find_security_patterns():
    """Find remaining security issues that need attention"""
    
    print("🔍 Searching for remaining security patterns...")
    
    # Patterns that might be legitimate but should be reviewed
    review_patterns = {
        'potential_secrets': [
            r'password\s*=\s*["\'][^"\']{3,}["\']',
            r'api_key\s*=\s*["\'][^"\']{10,}["\']',
            r'secret\s*=\s*["\'][^"\']{8,}["\']',
            r'token\s*=\s*["\'][^"\']{10,}["\']',
        ],
        'auth_related': [
            r'(password|secret|key|token)\s*[:=]\s*["\'][^"\']+["\']',
            r'AUTH_[A-Z_]+\s*=\s*["\'][^"\']+["\']',
        ],
        'config_files': [
            r'\.env',
            r'config\.py',
            r'settings\.py',
        ]
    }
    
    issues_found = []
    src_dirs = ['src', 'tests', 'deployment', '.']
    
    for src_dir in src_dirs:
        if not os.path.exists(src_dir):
            continue
            
        for root, dirs, files in os.walk(src_dir):
            # Skip version control and cache directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'venv']]
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                    
                filepath = os.path.join(root, file)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for category, patterns in review_patterns.items():
                            for pattern in patterns:
                                matches = re.finditer(pattern, content, re.IGNORECASE)
                                for match in matches:
                                    line_num = content[:match.start()].count('\n') + 1
                                    line_content = lines[line_num - 1].strip()
                                    
                                    # Skip comments and obvious test/example cases
                                    if (line_content.strip().startswith('#') or 
                                        'example' in line_content.lower() or
                                        'test' in filepath.lower() or
                                        'demo' in filepath.lower()):
                                        continue
                                        
                                    issues_found.append({
                                        'file': filepath,
                                        'line': line_num,
                                        'category': category,
                                        'content': line_content[:100],
                                        'pattern': pattern
                                    })
                        
                except Exception as e:
                    print(f"   Error reading {filepath}: {e}")
    
    return issues_found

def generate_security_report(issues):
    """Generate security report with recommendations"""
    
    if not issues:
        print("✅ No obvious hardcoded secrets found!")
        return
        
    print(f"\n🚨 Found {len(issues)} potential security issues to review:")
    print("=" * 60)
    
    by_category = {}
    for issue in issues:
        category = issue['category']
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(issue)
    
    for category, category_issues in by_category.items():
        print(f"\n📂 {category.upper()} ({len(category_issues)} issues):")
        for issue in category_issues[:10]:  # Show first 10
            print(f"   📄 {issue['file']}:{issue['line']}")
            print(f"      {issue['content']}")
        
        if len(category_issues) > 10:
            print(f"      ... and {len(category_issues) - 10} more")
    
    print(f"\n💡 SECURITY RECOMMENDATIONS:")
    print("=" * 60)
    print("1. Move all secrets to environment variables")
    print("2. Use .env files with proper .gitignore")
    print("3. Implement secure configuration management")
    print("4. Use secrets management services for production")
    print("5. Review each flagged line manually for actual secrets")

def create_secure_config_example():
    """Create example of secure configuration"""
    
    example_config = '''"""Secure configuration management example"""

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
'''
    
    with open('src/config/secure_config_example.py', 'w') as f:
        f.write(example_config)
    
    print("📝 Created secure configuration example at: src/config/secure_config_example.py")

if __name__ == "__main__":
    print("🔒 SECURITY AUDIT: Finding Remaining Issues")
    print("=" * 50)
    
    issues = find_security_patterns()
    generate_security_report(issues)
    
    # Create secure config directory if it doesn't exist
    os.makedirs('src/config', exist_ok=True)
    create_secure_config_example()
    
    print(f"\n✅ Security audit complete!")
    print("🔧 Next steps:")
    print("   1. Review flagged patterns manually")  
    print("   2. Move secrets to environment variables")
    print("   3. Use the secure configuration example")
    print("   4. Test with secure random generators")
    print("   5. Re-run security scan to verify fixes")