# Comprehensive PDF Uploader

A powerful PDF processing tool that extracts text, creates intelligent chunks, generates embeddings, and stores everything in ChromaDB for RAG applications.

## Features

- **Multiple PDF Readers**: Supports both PyPDF2 and pdfplumber
- **Intelligent Chunking**: Sentence-boundary aware chunking with configurable overlap
- **High-Quality Embeddings**: Uses Sentence Transformers for semantic embeddings
- **Batch Processing**: Memory-efficient batch processing for large documents
- **Comprehensive Metadata**: Rich metadata including keywords, page numbers, and statistics
- **Error Handling**: Robust error handling with detailed logging
- **Command Line Interface**: Easy-to-use CLI for quick processing
- **Programmatic API**: Full Python API for integration

## Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

```bash
# Basic usage
python pdf_uploader.py /path/to/document.pdf

# With custom settings
python pdf_uploader.py /path/to/document.pdf \
    --chunk-size 1500 \
    --overlap 300 \
    --collection-name "my_documents" \
    --embedding-model "all-mpnet-base-v2"
```

### Python API Usage

```python
from pdf_uploader import PDFUploader

# Initialize uploader
uploader = PDFUploader(
    chromadb_path="/path/to/chromadb",
    embedding_model="all-MiniLM-L6-v2"
)

# Process PDF
results = uploader.process_pdf(
    pdf_path="/path/to/document.pdf",
    chunk_size=1000,
    overlap=200,
    collection_name="my_collection"
)

if results['success']:
    print(f"Processed {results['statistics']['chunks_stored']} chunks")
```

## Configuration Options

### Embedding Models

Choose from various Sentence Transformer models:

- `all-MiniLM-L6-v2` (default): Fast, good quality, 384 dimensions
- `all-mpnet-base-v2`: Higher quality, slower, 768 dimensions
- `multi-qa-mpnet-base-dot-v1`: Optimized for Q&A tasks
- `all-distilroberta-v1`: Balanced speed/quality

### Chunking Parameters

- `chunk_size`: Target chunk size in characters (default: 1000)
- `overlap`: Overlap between chunks in characters (default: 200)

### ChromaDB Settings

- `chromadb_path`: Path to ChromaDB storage directory
- `collection_name`: Name for the ChromaDB collection (auto-generated if not specified)

## Command Line Options

```bash
python pdf_uploader.py --help
```

- `pdf_path`: Path to PDF file (required)
- `--chromadb-path`: ChromaDB storage directory
- `--embedding-model`: Sentence transformer model name
- `--collection-name`: ChromaDB collection name
- `--chunk-size`: Target chunk size in characters
- `--overlap`: Overlap between chunks
- `--verbose`: Enable verbose logging

## Examples

See `example_usage.py` for comprehensive examples:

```bash
python example_usage.py
```

## Output Format

The uploader provides detailed statistics:

```python
{
    'success': True,
    'pdf_path': '/path/to/document.pdf',
    'collection_name': 'document_collection',
    'statistics': {
        'pages_processed': 245,
        'chunks_created': 423,
        'chunks_embedded': 423,
        'chunks_stored': 423,
        'errors': []
    },
    'processing_time': '2024-01-15T10:30:00'
}
```

## Metadata Stored

Each chunk includes comprehensive metadata:

- Page number and position
- Character and word counts
- Sentence count
- Keywords extracted from text
- Creation timestamp
- Source PDF filename
- Chunk index

## Error Handling

The uploader handles various error conditions gracefully:

- Missing or corrupted PDF files
- Memory limitations during processing
- ChromaDB storage errors
- Embedding generation failures

All errors are logged and included in the results for debugging.

## Performance Tips

1. **Batch Size**: Large PDFs are processed in batches to manage memory
2. **Model Selection**: Choose embedding model based on speed vs. quality needs
3. **Chunk Size**: Smaller chunks = more granular search, larger chunks = more context
4. **Overlap**: More overlap = better context preservation, but more storage

## Integration with RAG Systems

The generated ChromaDB collection can be directly used with RAG systems:

```python
import chromadb
client = chromadb.PersistentClient(path="/path/to/chromadb")
collection = client.get_collection("my_collection")

# Search for similar content
results = collection.query(
    query_texts=["What is aerodynamics?"],
    n_results=5
)
```

## Troubleshooting

### Common Issues

1. **PDF Reading Errors**: Try switching between PyPDF2 and pdfplumber
2. **Memory Issues**: Reduce batch size or chunk size
3. **Embedding Errors**: Check internet connection for model download
4. **ChromaDB Errors**: Ensure write permissions to storage directory

### Logging

Enable verbose logging for debugging:

```bash
python pdf_uploader.py document.pdf --verbose
```