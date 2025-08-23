# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY advanced_rag_service.py .
COPY simple_pdf_uploader.py .
COPY PDF_UPLOADER_README.md .
COPY example_usage.py .

# Create ChromaDB data directory
RUN mkdir -p /app/chromadb_data

# Expose port
EXPOSE 8005

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=advanced_rag_service.py
ENV FLASK_ENV=production
ENV CHROMADB_PATH=/app/chromadb_data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Run the application
CMD ["python", "advanced_rag_service.py"]