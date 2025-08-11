# AI Science Platform - Deployment Guide

## 🚀 Deployment Overview

This guide provides step-by-step instructions for deploying the AI Science Platform in different environments, from development to production scale.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+ (recommended: 3.9+)
- **Memory**: Minimum 4GB RAM (8GB+ for production)
- **Storage**: 2GB available space (10GB+ for production with caching)
- **CPU**: Multi-core processor recommended for async processing

### Dependencies
- **Core**: numpy, asyncio (built-in)
- **Optional**: pytest (for testing), psutil (for system monitoring)
- **Development**: pytest-asyncio, coverage (for test coverage)

## 🛠️ Installation Methods

### Method 1: Automated Deployment (Recommended)

The platform includes an automated deployment script that handles all setup steps:

```bash
# Development deployment
python deploy.py development

# Production deployment  
python deploy.py production

# Skip tests during deployment (faster)
python deploy.py production --skip-tests

# Generate deployment report only
python deploy.py --report-only
```

### Method 2: Manual Installation

```bash
# 1. Clone/download the platform
git clone <repository-url>
cd ai-science-platform

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install platform in development mode
pip install -e .

# 5. Verify installation
python -m src.cli status
```

## 🌍 Environment Configurations

### Development Environment

**Purpose**: Local development, testing, and experimentation

**Configuration** (`.env.development`):
```bash
LOG_LEVEL=DEBUG
DISCOVERY_THRESHOLD=0.5
MAX_WORKERS=4
CACHE_ENABLED=true
AUTO_SCALING=false
BACKUP_ENABLED=false
HEALTH_CHECK_ENABLED=true
```

**Features Enabled**:
- ✅ Debug logging
- ✅ Lower discovery threshold for more results
- ✅ Limited workers for resource conservation
- ✅ Caching for performance
- ❌ Auto-scaling (manual control preferred)
- ❌ Backup system (not needed for development)

**Deployment Command**:
```bash
python deploy.py development
```

### Production Environment

**Purpose**: High-performance, scalable, reliable production deployment

**Configuration** (`.env.production`):
```bash
LOG_LEVEL=INFO
DISCOVERY_THRESHOLD=0.7
MAX_WORKERS=8
CACHE_ENABLED=true
AUTO_SCALING=true
BACKUP_ENABLED=true
HEALTH_CHECK_ENABLED=true
```

**Features Enabled**:
- ✅ Info-level logging (performance optimized)
- ✅ Higher discovery threshold for quality
- ✅ Maximum workers for throughput
- ✅ Multi-level caching system
- ✅ Auto-scaling for dynamic load handling
- ✅ Comprehensive backup system
- ✅ Real-time health monitoring

**Deployment Command**:
```bash
python deploy.py production
```

### Custom Environment

Create custom environment configurations by copying and modifying existing `.env` files:

```bash
# Copy production config as template
cp .env.production .env.custom

# Edit configuration
nano .env.custom

# Deploy with custom config
LOG_LEVEL=CUSTOM python deploy.py production
```

## 🔧 Advanced Deployment Options

### High-Performance Configuration

For maximum performance on powerful hardware:

```bash
# .env.high_performance
LOG_LEVEL=WARNING
DISCOVERY_THRESHOLD=0.8
MAX_WORKERS=16
CACHE_ENABLED=true
AUTO_SCALING=true
BACKUP_ENABLED=true
HEALTH_CHECK_ENABLED=true
CACHE_SIZE=5000
PERSISTENT_CACHE_SIZE_MB=2000
```

### Resource-Constrained Configuration

For deployment on limited hardware:

```bash
# .env.minimal
LOG_LEVEL=ERROR
DISCOVERY_THRESHOLD=0.6
MAX_WORKERS=2
CACHE_ENABLED=true
AUTO_SCALING=false
BACKUP_ENABLED=false
HEALTH_CHECK_ENABLED=false
CACHE_SIZE=100
```

## 🐳 Container Deployment

### Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY deploy.py .

# Install application
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs backups experiment_results data

# Expose port (if needed for future web interface)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.cli import main; import sys; sys.argv=['ai-science', 'status']; main()" || exit 1

# Default command
CMD ["python", "deploy.py", "production"]
```

**Build and run**:
```bash
# Build image
docker build -t ai-science-platform .

# Run container
docker run -d \
    --name ai-science \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/backups:/app/backups \
    ai-science-platform

# Check status
docker exec ai-science python -m src.cli status
```

### Kubernetes Deployment (Optional)

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-science-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-science-platform
  template:
    metadata:
      labels:
        app: ai-science-platform
    spec:
      containers:
      - name: ai-science
        image: ai-science-platform:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: MAX_WORKERS
          value: "8"
        - name: AUTO_SCALING
          value: "true"
        ports:
        - containerPort: 8000
        livenessProbe:
          exec:
            command:
            - python
            - -m
            - src.cli
            - status
          initialDelaySeconds: 30
          periodSeconds: 60
---
apiVersion: v1
kind: Service
metadata:
  name: ai-science-service
spec:
  selector:
    app: ai-science-platform
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

**Deploy**:
```bash
kubectl apply -f k8s-deployment.yaml
kubectl get pods -l app=ai-science-platform
```

## 🔍 Deployment Verification

### Automated Verification

The deployment script automatically performs verification:

```bash
# Comprehensive deployment with verification
python deploy.py production

# Output will show:
# ✅ Pre-deployment checks passed
# ✅ Dependencies installed
# ✅ Tests passed
# ✅ Package built
# ✅ Environment configured
# ✅ Deployment verified
# 📋 Deployment report generated
```

### Manual Verification

```bash
# 1. Check system status
python -m src.cli status

# 2. Test core functionality
python -c "
from src.algorithms.discovery import DiscoveryEngine
import numpy as np
engine = DiscoveryEngine()
data = np.random.randn(50, 4)
discoveries = engine.discover(data, context='verification')
print(f'Verification successful: {len(discoveries)} discoveries found')
"

# 3. Test CLI operations
python -m src.cli health

# 4. Check system resources
python -c "
from src.health_check import get_health_checker
health = get_health_checker().get_health_summary()
print(f'System status: {health[\"overall_status\"]}')
"
```

### Performance Validation

Run comprehensive performance validation:

```bash
# Full performance benchmark
python performance_validation.py

# Expected output:
# 🚀 Starting AI Science Platform Performance Validation
# 🔬 Validating discovery engine performance
# ⚡ Validating async processing performance
# 💾 Validating caching system performance
# 🛡️ Validating error handling performance
# 📈 Validating scalability features
# 🔒 Validating reliability features
# 🏥 Validating system health monitoring
# 🔗 Validating integration performance
# ✅ Performance validation completed!
#
# 🎯 Performance Validation Summary
# Overall Grade: A
# Overall Score: 92.3/100
```

## 📊 Monitoring & Maintenance

### Health Monitoring Setup

```bash
# Create monitoring script
cat > monitor_health.py << 'EOF'
#!/usr/bin/env python3
import time
import logging
from src.health_check import get_health_checker

logging.basicConfig(level=logging.INFO)

def monitor_system():
    checker = get_health_checker()
    
    while True:
        health = checker.get_health_summary()
        status = health['overall_status']
        
        if status != 'healthy':
            logging.warning(f"System health issue: {status}")
            # Add alerting logic here
        else:
            logging.info("System healthy")
        
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor_system()
EOF

# Run monitoring
python monitor_health.py &
```

### Log Rotation Setup

```bash
# Create log rotation configuration
cat > logrotate_ai_science << 'EOF'
/path/to/ai-science-platform/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF

# Install log rotation (Linux)
sudo cp logrotate_ai_science /etc/logrotate.d/
```

### Backup Configuration

```bash
# Create backup script
cat > backup_system.py << 'EOF'
#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
from datetime import datetime

def create_backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backups/system_backup_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup critical directories
    for directory in ['experiment_results', 'data', 'logs']:
        if Path(directory).exists():
            shutil.copytree(directory, backup_dir / directory)
    
    print(f"Backup created: {backup_dir}")

if __name__ == "__main__":
    create_backup()
EOF

# Schedule regular backups (cron example)
# 0 2 * * * cd /path/to/ai-science-platform && python backup_system.py
```

## 🚨 Troubleshooting

### Common Deployment Issues

#### 1. Python Version Issues
```bash
# Error: Python version too old
# Solution: Check Python version
python --version
# Must be 3.8+

# Update Python (Ubuntu/Debian)
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-pip
```

#### 2. Permission Errors
```bash
# Error: Permission denied writing to directories
# Solution: Create directories with proper permissions
mkdir -p logs backups experiment_results data
chmod 755 logs backups experiment_results data
```

#### 3. Memory Issues
```bash
# Error: Out of memory during large dataset processing
# Solution: Reduce MAX_WORKERS or enable memory-conscious settings
export MAX_WORKERS=2
export CACHE_SIZE=100
python deploy.py production
```

#### 4. Network/Firewall Issues
```bash
# Error: Connection timeouts
# Solution: Configure firewall for internal communication
# (Platform primarily operates locally, minimal network requirements)
```

### Performance Issues

#### Slow Discovery Performance
```bash
# Check system resources
python -c "
from src.health_check import get_health_checker
health = get_health_checker().get_health_summary()
print(f'CPU: {health[\"system_metrics\"][\"cpu_percent\"]}%')
print(f'Memory: {health[\"system_metrics\"][\"memory_percent\"]}%')
"

# Optimize configuration
export DISCOVERY_THRESHOLD=0.8  # Reduce discoveries for speed
export CACHE_ENABLED=true       # Enable caching
```

#### High Memory Usage
```bash
# Monitor memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / (1024**3):.1f}GB')
"

# Reduce memory footprint
export MAX_WORKERS=4
export CACHE_SIZE=500
export PERSISTENT_CACHE_SIZE_MB=100
```

### Recovery Procedures

#### System Recovery After Failure
```bash
# 1. Check system status
python -m src.cli status

# 2. Clear caches if corrupted
python -c "from src.performance.caching import clear_all_caches; clear_all_caches()"

# 3. Restart with clean state
rm -rf .cache/
python deploy.py production --skip-tests

# 4. Verify recovery
python performance_validation.py
```

#### Data Recovery
```bash
# List available backups
python -c "
from src.utils.backup import BackupManager
manager = BackupManager('backups/')
backups = manager.list_backups()
for backup in backups[-5:]:  # Show last 5 backups
    print(f'{backup[\"id\"]}: {backup[\"description\"]} ({backup[\"timestamp\"]})')
"

# Restore from backup
python -c "
from src.utils.backup import BackupManager
manager = BackupManager('backups/')
# Replace BACKUP_ID with actual backup ID
success = manager.restore_backup('BACKUP_ID', 'experiment_results/restored_data.txt')
print(f'Restore successful: {success}')
"
```

## 🔄 Upgrade Procedures

### Minor Updates
```bash
# 1. Backup current state
python -c "from src.utils.backup import BackupManager; BackupManager('backups/').create_backup('.', 'pre_upgrade')"

# 2. Update code
git pull origin main

# 3. Reinstall dependencies
pip install -r requirements.txt
pip install -e .

# 4. Redeploy
python deploy.py production

# 5. Verify upgrade
python -m src.cli status
python performance_validation.py
```

### Major Upgrades
```bash
# 1. Full system backup
tar -czf ai_science_backup_$(date +%Y%m%d).tar.gz .

# 2. Deploy to staging environment first
cp .env.production .env.staging
export MAX_WORKERS=2  # Reduced for staging
python deploy.py staging

# 3. Run comprehensive tests
python performance_validation.py

# 4. If successful, deploy to production
python deploy.py production
```

## 📈 Scaling Guidelines

### Vertical Scaling (Single Machine)
- **CPU**: Increase MAX_WORKERS proportionally to CPU cores
- **Memory**: Increase CACHE_SIZE and PERSISTENT_CACHE_SIZE_MB
- **Storage**: Expand backup and experiment results storage

### Horizontal Scaling (Multiple Machines)
- Deploy multiple instances with shared storage
- Use load balancer for discovery requests
- Coordinate caching across instances
- Implement distributed backup strategy

### Auto-Scaling Configuration
```bash
# Enable auto-scaling with custom thresholds
export AUTO_SCALING=true
export MIN_WORKERS=2
export MAX_WORKERS=16
export SCALE_UP_THRESHOLD=80    # CPU percentage
export SCALE_DOWN_THRESHOLD=20
export SCALE_INTERVAL=30        # Seconds between scaling decisions
```

---

This deployment guide provides comprehensive instructions for deploying the AI Science Platform across different environments and scenarios. Follow the automated deployment method for the quickest setup, or use the manual procedures for custom configurations.

**🤖 Generated with Claude Code**
**Co-Authored-By: Claude <noreply@anthropic.com>**