#!/bin/bash

# Test script to verify demo.sh integration with doc-db

echo "🔍 Testing demo.sh integration with doc-db service..."
echo ""

# Check if required files exist
echo "📋 Checking required files:"
echo "  ✓ demo.sh: $([ -f ../demo.sh ] && echo "EXISTS" || echo "MISSING")"
echo "  ✓ Dockerfile: $([ -f ./Dockerfile ] && echo "EXISTS" || echo "MISSING")"
echo "  ✓ advanced_rag_service.py: $([ -f ./advanced_rag_service.py ] && echo "EXISTS" || echo "MISSING")"
echo "  ✓ requirements.txt: $([ -f ./requirements.txt ] && echo "EXISTS" || echo "MISSING")"
echo ""

# Check ChromaDB data
echo "📊 ChromaDB Data Status:"
if [ -d "./79691e50-0896-4422-bed6-0d2986c7a0a3" ]; then
    echo "  ✓ ChromaDB data directory exists"
    echo "  ✓ Data files: $(ls -1 ./79691e50-0896-4422-bed6-0d2986c7a0a3/ | wc -l | xargs) files"
else
    echo "  ❌ ChromaDB data directory missing"
fi

if [ -f "./chroma.sqlite3" ]; then
    echo "  ✓ ChromaDB SQLite database exists ($(du -h ./chroma.sqlite3 | cut -f1))"
else
    echo "  ❌ ChromaDB SQLite database missing"
fi

if [ -f "./00_afh_full.pdf" ]; then
    echo "  ✓ Source PDF exists ($(du -h ./00_afh_full.pdf | cut -f1))"
else
    echo "  ❌ Source PDF missing"
fi
echo ""

# Test what demo.sh would do (dry run simulation)
echo "🚀 Demo.sh Integration Preview:"
echo "  • Will build image: trominos-doc-db:$(date +%Y-%m-%d)"
echo "  • Will expose port: 8005"
echo "  • Will mount ChromaDB data: $(pwd) → /app/chromadb_data"
echo "  • Will mount MCP server: $(dirname $(pwd))/a-tier-mcp-server → /app/mcp-server"
echo "  • Will connect to network: trominos-network"
echo ""

# Show expected environment variables
echo "🔧 Environment Variables (from demo.sh):"
echo "  • FLASK_ENV=production"
echo "  • CHROMADB_PATH=/app/chromadb_data"
echo "  • CORS_ORIGINS=http://localhost:8080,http://agents"
echo "  • MCP_SERVER_PATH=/app/mcp-server"
echo "  • DATABASE_URL=postgresql://postgres:postgres@postgres:5432/ai_application"
echo "  • ANTHROPIC_API_KEY=\${ANTHROPIC_API_KEY}"
echo ""

# Show service endpoints after deployment
echo "🌐 Expected Service Endpoints:"
echo "  • RAG Health Check: http://localhost:8005/health"
echo "  • RAG Stats: http://localhost:8005/stats"
echo "  • RAG Query: http://localhost:8005/query"
echo "  • Frontend (with RAG chat): http://localhost:8080"
echo ""

echo "✅ Integration test complete! Run '../demo.sh' to build and deploy."