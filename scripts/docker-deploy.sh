#!/bin/bash
# Docker deployment script for AI Science Platform

set -e

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="ai-science-platform"
ENV_FILE=".env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AI Science Platform Deployment${NC}"
echo "=================================="

# Parse command line arguments
ENVIRONMENT="production"
ACTION="up"
BUILD=false
DETACH=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            ENVIRONMENT="development"
            COMPOSE_FILE="docker-compose.dev.yml"
            shift
            ;;
        --prod)
            ENVIRONMENT="production"
            COMPOSE_FILE="docker-compose.yml"
            shift
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --foreground)
            DETACH=false
            shift
            ;;
        up|down|restart|logs|status)
            ACTION="$1"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options] [action]"
            echo ""
            echo "Actions:"
            echo "  up       Start services (default)"
            echo "  down     Stop and remove services"
            echo "  restart  Restart services"
            echo "  logs     Show logs"
            echo "  status   Show service status"
            echo ""
            echo "Options:"
            echo "  --dev        Deploy in development mode"
            echo "  --prod       Deploy in production mode (default)"
            echo "  --build      Build images before deploying"
            echo "  --foreground Run in foreground (don't detach)"
            echo "  -h, --help   Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Configuration:${NC}"
echo "  Environment: $ENVIRONMENT"
echo "  Compose File: $COMPOSE_FILE"
echo "  Action: $ACTION"
echo "  Build: $BUILD"
echo ""

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker.${NC}"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose.${NC}"
    exit 1
fi

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}❌ Compose file not found: $COMPOSE_FILE${NC}"
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f "$ENV_FILE" ] && [ "$ENVIRONMENT" = "production" ]; then
    echo -e "${YELLOW}⚠️ Creating default .env file${NC}"
    cat > $ENV_FILE << EOF
# AI Science Platform Environment Variables
VERSION=latest
LOG_LEVEL=INFO
WORKERS=4
SECRET_KEY=your-secret-key-change-this-in-production
POSTGRES_DB=ai_science
POSTGRES_USER=ai_user  
POSTGRES_PASSWORD=your-secure-password-here
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
ENVIRONMENT=$ENVIRONMENT
EOF
    echo -e "${YELLOW}⚠️ Please update the .env file with secure values before production deployment${NC}"
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"
echo ""

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p data logs cache models ssl

# Set proper permissions
if [ "$ENVIRONMENT" = "production" ]; then
    sudo chown -R 1000:1000 data logs cache models 2>/dev/null || true
fi

echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Execute the requested action
case $ACTION in
    "up")
        echo -e "${BLUE}🚀 Starting services...${NC}"
        
        BUILD_FLAG=""
        if [ "$BUILD" = true ]; then
            BUILD_FLAG="--build"
        fi
        
        DETACH_FLAG="-d"
        if [ "$DETACH" = false ]; then
            DETACH_FLAG=""
        fi
        
        if command -v docker-compose &> /dev/null; then
            docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up $BUILD_FLAG $DETACH_FLAG
        else
            docker compose -f $COMPOSE_FILE -p $PROJECT_NAME up $BUILD_FLAG $DETACH_FLAG
        fi
        
        if [ "$DETACH" = true ]; then
            echo -e "${GREEN}✅ Services started in background${NC}"
            echo ""
            echo -e "${BLUE}📊 Service Status:${NC}"
            if command -v docker-compose &> /dev/null; then
                docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
            else
                docker compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
            fi
        fi
        ;;
        
    "down")
        echo -e "${BLUE}🛑 Stopping services...${NC}"
        
        if command -v docker-compose &> /dev/null; then
            docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down
        else
            docker compose -f $COMPOSE_FILE -p $PROJECT_NAME down
        fi
        
        echo -e "${GREEN}✅ Services stopped${NC}"
        ;;
        
    "restart")
        echo -e "${BLUE}🔄 Restarting services...${NC}"
        
        if command -v docker-compose &> /dev/null; then
            docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME restart
        else
            docker compose -f $COMPOSE_FILE -p $PROJECT_NAME restart
        fi
        
        echo -e "${GREEN}✅ Services restarted${NC}"
        ;;
        
    "logs")
        echo -e "${BLUE}📄 Showing logs...${NC}"
        
        if command -v docker-compose &> /dev/null; then
            docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f
        else
            docker compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f
        fi
        ;;
        
    "status")
        echo -e "${BLUE}📊 Service Status:${NC}"
        
        if command -v docker-compose &> /dev/null; then
            docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
        else
            docker compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
        fi
        
        echo ""
        echo -e "${BLUE}💾 Volume Usage:${NC}"
        docker system df
        ;;
esac

echo ""
echo -e "${BLUE}🔗 Access URLs:${NC}"
if [ "$ENVIRONMENT" = "development" ]; then
    echo "  Application: http://localhost:8000"
    echo "  Jupyter Lab: http://localhost:8888"
    echo "  Redis: localhost:6380"
else
    echo "  Application: http://localhost:8000"
    echo "  HTTPS: https://localhost (if SSL configured)"
    echo "  Database: localhost:5432"
    echo "  Redis: localhost:6379"
fi

echo ""
echo -e "${GREEN}🎉 Deployment process completed!${NC}"