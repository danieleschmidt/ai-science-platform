#!/bin/bash
# Docker build script for AI Science Platform

set -e

# Configuration
IMAGE_NAME="terragon/ai-science-platform"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION=${VERSION:-latest}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 AI Science Platform Docker Build${NC}"
echo "=================================="

# Parse command line arguments
TARGET="production"
PUSH=false
NO_CACHE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            TARGET="development"
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --dev        Build development image"
            echo "  --push       Push to registry after build"
            echo "  --no-cache   Build without cache"
            echo "  --version    Set version tag"
            echo "  -h, --help   Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Set image tag based on target
if [ "$TARGET" = "development" ]; then
    IMAGE_TAG="${IMAGE_NAME}:dev"
    DOCKERFILE="Dockerfile.dev"
else
    IMAGE_TAG="${IMAGE_NAME}:${VERSION}"
    DOCKERFILE="Dockerfile"
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Target: $TARGET"
echo "  Image: $IMAGE_TAG"
echo "  Dockerfile: $DOCKERFILE"
echo "  Build Date: $BUILD_DATE"
echo "  VCS Ref: $VCS_REF"
echo "  Version: $VERSION"
echo ""

# Pre-build checks
echo -e "${BLUE}🔍 Pre-build checks...${NC}"

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}❌ Dockerfile not found: $DOCKERFILE${NC}"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
    echo -e "${YELLOW}⚠️ No setup.py or pyproject.toml found. Are you in the project root?${NC}"
fi

echo -e "${GREEN}✅ Pre-build checks passed${NC}"
echo ""

# Build the image
echo -e "${BLUE}🔨 Building Docker image...${NC}"

BUILD_ARGS="--build-arg BUILD_DATE=$BUILD_DATE --build-arg VERSION=$VERSION --build-arg VCS_REF=$VCS_REF"

if [ "$NO_CACHE" = true ]; then
    BUILD_ARGS="$BUILD_ARGS --no-cache"
fi

# Execute build
docker build $BUILD_ARGS -f $DOCKERFILE -t $IMAGE_TAG .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Build completed successfully${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

# Tag with additional tags if needed
if [ "$TARGET" = "production" ] && [ "$VERSION" != "latest" ]; then
    docker tag $IMAGE_TAG "${IMAGE_NAME}:latest"
    echo -e "${GREEN}✅ Tagged as latest${NC}"
fi

# Display image info
echo -e "${BLUE}📋 Image Information:${NC}"
docker images $IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}"

# Test the image
echo -e "${BLUE}🧪 Testing image...${NC}"
if docker run --rm $IMAGE_TAG python -c "import src; print('✅ Import test passed')"; then
    echo -e "${GREEN}✅ Image test passed${NC}"
else
    echo -e "${RED}❌ Image test failed${NC}"
    exit 1
fi

# Security scan (if available)
if command -v docker scan &> /dev/null; then
    echo -e "${BLUE}🔒 Running security scan...${NC}"
    docker scan $IMAGE_TAG || echo -e "${YELLOW}⚠️ Security scan completed with warnings${NC}"
fi

# Push if requested
if [ "$PUSH" = true ]; then
    echo -e "${BLUE}📤 Pushing to registry...${NC}"
    docker push $IMAGE_TAG
    
    if [ "$TARGET" = "production" ] && [ "$VERSION" != "latest" ]; then
        docker push "${IMAGE_NAME}:latest"
    fi
    
    echo -e "${GREEN}✅ Push completed${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Docker build process completed successfully!${NC}"
echo ""
echo -e "${BLUE}Usage examples:${NC}"
echo "  Run container: docker run -p 8000:8000 $IMAGE_TAG"
echo "  With docker-compose: docker-compose up"
echo "  Development mode: docker-compose -f docker-compose.dev.yml up"
echo ""