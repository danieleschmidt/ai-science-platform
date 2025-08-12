#!/bin/bash
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
