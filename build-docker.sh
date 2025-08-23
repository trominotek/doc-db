#!/bin/bash

# Build script for doc-db Docker image

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building doc-db Docker image...${NC}"

# Get the current date for tagging
DATE_TAG=$(date +%Y-%m-%d)
LATEST_TAG="latest"

# Build the Docker image
docker build -t trominos-doc-db:${DATE_TAG} -t trominos-doc-db:${LATEST_TAG} .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully!${NC}"
    echo -e "${BLUE}Images created:${NC}"
    echo "  - trominos-doc-db:${DATE_TAG}"
    echo "  - trominos-doc-db:${LATEST_TAG}"
    echo ""
    echo -e "${BLUE}To run the container:${NC}"
    echo "docker run -p 8005:8005 -v /Users/tojojose/trominos/doc-db:/app/chromadb_data trominos-doc-db:${LATEST_TAG}"
    echo ""
    echo -e "${BLUE}Or use docker-compose:${NC}"
    echo "docker-compose up doc-db"
else
    echo -e "${RED}❌ Docker image build failed!${NC}"
    exit 1
fi