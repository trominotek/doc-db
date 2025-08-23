# Doc-DB Docker Configuration

This directory contains the Docker configuration for the doc-db RAG (Retrieval-Augmented Generation) service.

## 🐳 Docker Setup

### Files Created
- `Dockerfile` - Container definition for the doc-db service
- `build-docker.sh` - Build script for creating the Docker image
- `.dockerignore` - Excludes unnecessary files from build context
- `DOCKER_README.md` - This documentation

## 🚀 Quick Start

### 1. Build the Docker Image
```bash
# From the doc-db directory
./build-docker.sh
```

### 2. Run with Docker Compose (Recommended)
```bash
# From the project root directory
docker-compose up doc-db
```

### 3. Run Standalone Container
```bash
docker run -d \
  --name doc-db-rag \
  -p 8005:8005 \
  -v /Users/tojojose/trominos/doc-db:/app/chromadb_data:rw \
  -v /Users/tojojose/trominos/a-tier-mcp-server:/app/mcp-server:ro \
  -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
  trominos-doc-db:latest
```

## 📋 Docker Compose Configuration

The doc-db service is integrated into the main `docker-compose.yml` with the following configuration:

```yaml
doc-db:
  image: trominos-doc-db:2025-08-21
  ports:
    - "8005:8005"
  environment:
    - FLASK_ENV=production
    - CHROMADB_PATH=/app/chromadb_data
    - CORS_ORIGINS=http://localhost:8080,http://agents
    - MCP_SERVER_PATH=/app/mcp-server
    - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/ai_application
    - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
  volumes:
    # Preserves existing ChromaDB data
    - /Users/tojojose/trominos/doc-db:/app/chromadb_data:rw
    # Access to MCP server for API key management
    - /Users/tojojose/trominos/a-tier-mcp-server:/app/mcp-server:ro
  depends_on:
    postgres:
      condition: service_healthy
  healthcheck:
    test: ["CMD-SHELL", "curl -f http://localhost:8005/health || exit 1"]
    interval: 30s
    timeout: 10s
    retries: 3
```

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `production` | Flask environment (production/development) |
| `PORT` | `8005` | Port for the Flask application |
| `CHROMADB_PATH` | `/app/chromadb_data` | Path to ChromaDB storage |
| `CORS_ORIGINS` | `http://localhost:8080,http://agents` | Allowed CORS origins |
| `MCP_SERVER_PATH` | `/app/mcp-server` | Path to MCP server code |
| `ANTHROPIC_API_KEY` | - | Anthropic API key for Claude integration |
| `DATABASE_URL` | - | PostgreSQL connection string |

## 📁 Volume Mounts

### ChromaDB Data Volume
- **Host Path:** `/Users/tojojose/trominos/doc-db`
- **Container Path:** `/app/chromadb_data`
- **Mode:** Read/Write (`rw`)
- **Purpose:** Preserves existing ChromaDB database with 399 aviation handbook chunks

### MCP Server Volume
- **Host Path:** `/Users/tojojose/trominos/a-tier-mcp-server`
- **Container Path:** `/app/mcp-server`
- **Mode:** Read-Only (`ro`)
- **Purpose:** Access to API key management system

## 🌐 Network Configuration

The service runs on the `trominos-network` bridge network and communicates with:
- **Agents UI** (port 8080) - Frontend for the chat interface
- **PostgreSQL** (port 5432) - Database for API key storage
- **MCP Server** (port 8000) - API key management

## 🏥 Health Checks

The container includes a health check that:
- Tests the `/health` endpoint every 30 seconds
- Has a 10-second timeout
- Retries 3 times before marking as unhealthy
- Provides detailed service status information

## 📊 Service APIs

The containerized service exposes these endpoints:

- `GET /health` - Health check and service status
- `GET /stats` - ChromaDB statistics (documents, chunks, model info)
- `POST /query` - Main RAG query endpoint for chat functionality
- `POST /query/rag` - Advanced RAG queries with session management
- `POST /quiz/generate` - Generate training quizzes
- `GET /vector-store/info` - Vector store information
- `GET /collections/list` - List available collections

## 🔍 Troubleshooting

### Check Service Status
```bash
# Check if container is running
docker-compose ps doc-db

# Check container logs
docker-compose logs doc-db

# Check health status
docker-compose exec doc-db curl http://localhost:8005/health
```

### Common Issues

1. **Port Conflicts**
   - Ensure port 8005 is not in use on the host
   - Check with: `lsof -i :8005`

2. **Volume Mount Issues**
   - Verify ChromaDB data exists at `/Users/tojojose/trominos/doc-db`
   - Check directory permissions

3. **API Key Issues**
   - Ensure `ANTHROPIC_API_KEY` environment variable is set
   - Check MCP server connection for database-stored keys

4. **CORS Issues**
   - Update `CORS_ORIGINS` environment variable if accessing from different domains
   - Default allows `localhost:8080` and internal `agents` service

### Rebuild Container
```bash
# Stop existing container
docker-compose down doc-db

# Rebuild and start
./build-docker.sh
docker-compose up doc-db
```

## 📈 Performance Notes

- **ChromaDB Data:** Preserved across container restarts via volume mount
- **No Re-uploading:** Existing 399 aviation handbook chunks are maintained
- **Memory Usage:** Optimized for production with minimal ML dependencies
- **Startup Time:** Fast startup since data is pre-loaded

## 🔒 Security

- MCP server mounted read-only for security
- Environment variables for sensitive configuration
- CORS properly configured for allowed origins
- Health checks for service monitoring

## 🚀 Production Deployment

For production deployment:

1. Update image tags in `docker-compose.yml`
2. Set proper environment variables
3. Configure external volumes for data persistence
4. Set up proper logging and monitoring
5. Use production CORS origins
6. Consider using Docker secrets for API keys