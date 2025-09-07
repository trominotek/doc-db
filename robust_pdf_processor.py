#!/usr/bin/env python3
"""
Robust PDF Processing Service for Doc-DB
Based on the rag-quiz approach with proper error handling and ChromaDB integration
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import PyPDF2
import pdfplumber

class DocDBConfig:
    """Configuration for doc-db PDF processing"""
    
    def __init__(self):
        # Detect if running in Docker container or locally
        if Path("/app").exists() and Path("/app").is_dir():
            # Docker container environment
            self.BASE_DIR = Path("/app")
        else:
            # Local development environment
            self.BASE_DIR = Path(__file__).parent
        
        self.PDF_DIR = self.BASE_DIR / "pdfs"
        if Path("/app").exists():
            # Docker container - use chromadb_data subdirectory
            self.CHROMA_DIR = self.BASE_DIR / "chromadb_data"
        else:
            # Local development - use rag-quiz ChromaDB directory
            self.CHROMA_DIR = self.BASE_DIR.parent / "rag-quiz" / "db" / "chroma"
        self.PROCESSED_DATA_DIR = self.BASE_DIR / "processed_data"
        
        # Ensure directories exist (safe creation)
        try:
            self.PDF_DIR.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            pass  # Directory may already exist or be read-only
            
        try:
            if not self.CHROMA_DIR.exists():
                self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            pass  # Directory may already exist or be read-only
            
        try:
            self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            pass  # Directory may already exist or be read-only
        
        # Model settings
        self.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Chunking settings
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
    
    def get_pdf_files(self):
        """Get list of PDF files"""
        return list(self.PDF_DIR.glob("*.pdf"))

class RobustChromaDBManager:
    """Robust ChromaDB manager based on rag-quiz approach"""
    
    def __init__(self, config: DocDBConfig, existing_client=None):
        self.config = config
        self.client = existing_client
        self.collection = None
        self.embedding_function = None
        if not self.client:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with proper error handling"""
        try:
            # Try to reuse existing client or create new one
            self.client = chromadb.PersistentClient(
                path=str(self.config.CHROMA_DIR)
            )
            print(f"ChromaDB client initialized at {self.config.CHROMA_DIR}")
            
            # Set up embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.EMBEDDING_MODEL
            )
            print(f"Embedding function initialized with model: {self.config.EMBEDDING_MODEL}")
            
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise
    
    def create_collection(self, collection_name: str = "rag_documents", reset: bool = False) -> bool:
        """Create or get a collection"""
        try:
            if reset:
                # Delete existing collection if it exists
                try:
                    self.client.delete_collection(collection_name)
                    print(f"Deleted existing collection: {collection_name}")
                except Exception as e:
                    print(f"No existing collection to delete: {e}")
                    pass
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "RAG document chunks with embeddings"}
            )
            
            print(f"Collection '{collection_name}' ready")
            return True
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
    
    def clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata for ChromaDB compatibility"""
        cleaned = {}
        
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = None
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                cleaned[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                # Convert dicts to JSON strings
                cleaned[key] = str(value)
            else:
                # Convert other types to strings
                cleaned[key] = str(value)
        
        return cleaned
    
    def add_chunks(self, chunks: List[Dict], batch_size: int = 100) -> bool:
        """Add chunks to the collection"""
        if not self.collection:
            print("No collection available. Create a collection first.")
            return False
        
        try:
            total_chunks = len(chunks)
            print(f"Adding {total_chunks} chunks to ChromaDB...")
            
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Prepare data for ChromaDB
                ids = [chunk['id'] for chunk in batch_chunks]
                documents = [chunk['text'] for chunk in batch_chunks]
                # Clean metadata to ensure ChromaDB compatibility
                metadatas = [self.clean_metadata(chunk['metadata']) for chunk in batch_chunks]
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                
                print(f"Added batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
            
            print(f"Successfully added {total_chunks} chunks to ChromaDB")
            return True
            
        except Exception as e:
            print(f"Error adding chunks to ChromaDB: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            # Try to get the rag_documents collection
            if not self.collection:
                try:
                    self.collection = self.client.get_collection("rag_documents")
                except:
                    return {'name': 'N/A', 'count': 0}
            
            count = self.collection.count()
            return {
                'name': self.collection.name,
                'count': count,
                'metadata': self.collection.metadata
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {'name': 'Error', 'count': 0}

class RobustPDFProcessor:
    """Robust PDF processor based on rag-quiz approach"""
    
    def __init__(self, chroma_client=None):
        self.config = DocDBConfig()
        self.chroma_manager = RobustChromaDBManager(self.config, chroma_client)
        
        self.processing_status = {
            'status': 'idle',
            'current_file': None,
            'progress': 0,
            'total_files': 0,
            'message': 'Ready to process documents',
            'last_updated': datetime.now().isoformat()
        }
    
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files safely"""
        try:
            pdf_files = list(self.config.PDF_DIR.glob("*.pdf"))
            return sorted(pdf_files)
        except Exception as e:
            print(f"Error listing PDF files: {e}")
            return []
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        return self.processing_status.copy()
    
    def update_status(self, status: str, message: str, current_file: str = None, progress: int = None):
        """Update processing status"""
        self.processing_status.update({
            'status': status,
            'message': message,
            'last_updated': datetime.now().isoformat()
        })
        if current_file is not None:
            self.processing_status['current_file'] = current_file
        if progress is not None:
            self.processing_status['progress'] = progress
        
        print(f"[{status.upper()}] {message}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF with fallback methods"""
        text = ""
        
        try:
            # Try with pdfplumber first (better for complex layouts)
            print(f"Extracting text from {pdf_path.name} using pdfplumber")
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"[Page {page_num + 1}]\n{page_text}\n\n"
                    except Exception as e:
                        print(f"Error extracting page {page_num + 1}: {e}")
                        continue
                        
        except Exception as e:
            print(f"pdfplumber failed for {pdf_path.name}, trying PyPDF2: {e}")
            
            try:
                # Fallback to PyPDF2
                print(f"Extracting text from {pdf_path.name} using PyPDF2")
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += f"[Page {page_num + 1}]\n{page_text}\n\n"
                        except Exception as e:
                            print(f"Error extracting page {page_num + 1}: {e}")
                            continue
                            
            except Exception as e2:
                print(f"Both PDF extraction methods failed for {pdf_path.name}: {e2}")
                return ""
        
        return text.strip()
    
    def create_chunks(self, text: str, filename: str, file_path: str) -> List[Dict[str, Any]]:
        """Create chunks from text with metadata"""
        chunks = []
        
        # Split text into sections by pages if available
        sections = text.split('[Page ')
        current_section = ""
        current_page = 1
        
        for i, section in enumerate(sections):
            if i == 0 and not section.startswith('1]'):
                # First section without page marker
                current_section = section
                continue
                
            if section.strip():
                # Extract page number and content
                if ']' in section:
                    page_part, content = section.split(']', 1)
                    try:
                        current_page = int(page_part.strip())
                    except:
                        current_page = i
                    current_section = content.strip()
                else:
                    current_section = section.strip()
                
                # Chunk the section text
                words = current_section.split()
                
                for j in range(0, len(words), self.config.CHUNK_SIZE - self.config.CHUNK_OVERLAP):
                    chunk_words = words[j:j + self.config.CHUNK_SIZE]
                    chunk_text = ' '.join(chunk_words)
                    
                    if chunk_text.strip() and len(chunk_text.strip()) > 50:  # Minimum chunk size
                        chunk_id = f"{filename}_{current_page}_{j//100}_{uuid.uuid4().hex[:8]}"
                        chunks.append({
                            'id': chunk_id,
                            'text': chunk_text.strip(),
                            'metadata': {
                                'source_file': filename,
                                'source_path': file_path,
                                'page_number': str(current_page),
                                'chunk_index': len(chunks),
                                'word_count': len(chunk_words),
                                'char_count': len(chunk_text),
                                'created_at': datetime.now().isoformat(),
                                'document_type': 'pdf'
                            }
                        })
        
        return chunks
    
    def process_all_pdfs(self, reset_collection: bool = True) -> Dict[str, Any]:
        """Process all PDFs with robust error handling"""
        try:
            pdf_files = self.get_pdf_files()
            
            if not pdf_files:
                self.update_status('completed', 'No PDF files found in pdfs/ folder')
                return {
                    'success': False,
                    'message': 'No PDF files found',
                    'files_processed': 0
                }
            
            self.processing_status['total_files'] = len(pdf_files)
            self.update_status('processing', f'Starting processing of {len(pdf_files)} PDF files', progress=0)
            
            # Step 1: Create collection
            self.update_status('processing', 'Initializing vector database...', progress=5)
            if not self.chroma_manager.create_collection("rag_documents", reset=reset_collection):
                return {
                    'success': False,
                    'message': 'Failed to create ChromaDB collection',
                    'files_processed': 0
                }
            
            all_chunks = []
            processed_files = []
            
            # Step 2: Process each PDF
            for i, pdf_file in enumerate(pdf_files):
                progress = 10 + (i * 70 // len(pdf_files))
                self.update_status('processing', f'Processing: {pdf_file.name}', 
                                 current_file=pdf_file.name, progress=progress)
                
                # Extract text
                text = self.extract_text_from_pdf(pdf_file)
                if not text:
                    print(f"No text extracted from {pdf_file.name}, skipping")
                    continue
                
                # Create chunks
                chunks = self.create_chunks(text, pdf_file.name, str(pdf_file))
                if chunks:
                    all_chunks.extend(chunks)
                    processed_files.append(pdf_file.name)
                    print(f"Created {len(chunks)} chunks from {pdf_file.name}")
            
            if not all_chunks:
                self.update_status('error', 'No content could be extracted from PDF files')
                return {
                    'success': False,
                    'message': 'No content extracted from PDF files',
                    'files_processed': len(processed_files)
                }
            
            # Step 3: Add chunks to ChromaDB
            self.update_status('processing', 'Adding chunks to vector database...', progress=85)
            
            if not self.chroma_manager.add_chunks(all_chunks):
                return {
                    'success': False,
                    'message': 'Failed to add chunks to ChromaDB',
                    'files_processed': len(processed_files)
                }
            
            # Step 4: Verify and save metadata
            collection_info = self.chroma_manager.get_collection_info()
            self.save_processing_metadata(all_chunks, collection_info, processed_files)
            
            self.update_status('completed', 
                             f'Successfully processed {len(processed_files)} files, {len(all_chunks)} chunks created',
                             progress=100)
            
            return {
                'success': True,
                'message': 'PDF processing completed successfully',
                'files_processed': len(processed_files),
                'total_chunks': len(all_chunks),
                'collection_info': collection_info
            }
            
        except Exception as e:
            error_msg = f'Error during PDF processing: {str(e)}'
            self.update_status('error', error_msg)
            return {
                'success': False,
                'message': error_msg,
                'files_processed': 0
            }
    
    def save_processing_metadata(self, chunks: List[Dict], collection_info: Dict[str, Any], files_processed: List[str]):
        """Save processing metadata"""
        try:
            metadata = {
                'processing_timestamp': datetime.now().isoformat(),
                'config': {
                    'chunk_size': self.config.CHUNK_SIZE,
                    'chunk_overlap': self.config.CHUNK_OVERLAP,
                    'embedding_model': self.config.EMBEDDING_MODEL,
                },
                'statistics': {
                    'total_chunks': len(chunks),
                    'total_characters': sum(chunk['metadata']['char_count'] for chunk in chunks),
                    'total_words': sum(chunk['metadata']['word_count'] for chunk in chunks),
                    'avg_chunk_size': sum(chunk['metadata']['char_count'] for chunk in chunks) / len(chunks) if chunks else 0,
                    'files_processed': files_processed,
                    'collection_info': collection_info
                }
            }
            
            metadata_path = self.config.BASE_DIR / "processing_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"Processing metadata saved to {metadata_path}")
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics safely"""
        try:
            collection_info = self.chroma_manager.get_collection_info()
            
            # Try to load processing metadata
            metadata_path = self.config.BASE_DIR / "processing_metadata.json"
            processing_metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        processing_metadata = json.load(f)
                except Exception as e:
                    print(f"Error loading metadata: {e}")
            
            return {
                'collection_info': collection_info,
                'processing_metadata': processing_metadata,
                'pdf_files': [f.name for f in self.get_pdf_files()],
                'last_processing': processing_metadata.get('processing_timestamp', 'Never')
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {
                'error': str(e),
                'collection_info': {'name': 'Error', 'count': 0},
                'processing_metadata': {},
                'pdf_files': [f.name for f in self.get_pdf_files()],
                'last_processing': 'Error retrieving information'
            }

def main():
    """Test the processor"""
    processor = RobustPDFProcessor()
    
    print("📄 PDF Files found:")
    for pdf in processor.get_pdf_files():
        print(f"  • {pdf.name}")
    
    print("\n📊 Collection Stats:")
    stats = processor.get_collection_stats()
    print(f"  Collection: {stats['collection_info']['name']}")
    print(f"  Chunks: {stats['collection_info']['count']}")
    print(f"  Last processing: {stats['last_processing']}")

if __name__ == "__main__":
    main()