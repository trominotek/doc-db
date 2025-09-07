#!/usr/bin/env python3
"""
Simple PDF Processing Service for Doc-DB
Simplified version that works within the Docker container
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import pdfplumber

class SimplePDFProcessor:
    """Simple PDF processor that works in the Docker container"""
    
    def __init__(self):
        self.base_dir = Path("/app")
        self.pdfs_dir = self.base_dir / "pdfs"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Create directories
        self.pdfs_dir.mkdir(exist_ok=True)
        
        self.processing_status = {
            'status': 'idle',
            'current_file': None,
            'progress': 0,
            'total_files': 0,
            'message': 'Ready to process documents',
            'last_updated': datetime.now().isoformat()
        }
    
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files in the pdfs directory"""
        pdf_files = list(self.pdfs_dir.glob("*.pdf"))
        return sorted(pdf_files)
    
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
        """Extract text from a PDF file"""
        text = ""
        
        try:
            # Try with pdfplumber first
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber failed for {pdf_path.name}, trying PyPDF2: {e}")
            
            try:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e2:
                print(f"PyPDF2 also failed for {pdf_path.name}: {e2}")
                return ""
        
        return text.strip()
    
    def chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Simple text chunking"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                chunk_id = f"{filename}_{i//self.chunk_size}_{uuid.uuid4().hex[:8]}"
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'metadata': {
                        'source_file': filename,
                        'chunk_index': len(chunks),
                        'word_count': len(chunk_words),
                        'char_count': len(chunk_text)
                    }
                })
        
        return chunks
    
    def process_all_pdfs(self, reset_collection: bool = True) -> Dict[str, Any]:
        """Process all PDFs in the pdfs folder"""
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
            
            # Initialize ChromaDB
            chroma_client = chromadb.PersistentClient(path="/app/chromadb_data")
            
            if reset_collection:
                try:
                    chroma_client.delete_collection("rag_documents")
                except:
                    pass
            
            # Create collection
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            
            collection = chroma_client.get_or_create_collection(
                name="rag_documents",
                embedding_function=embedding_function,
                metadata={"description": "RAG document chunks with embeddings"}
            )
            
            all_chunks = []
            
            # Process each PDF
            for i, pdf_file in enumerate(pdf_files):
                progress = 20 + (i * 60 // len(pdf_files))
                self.update_status('processing', f'Extracting text from: {pdf_file.name}', 
                                 current_file=pdf_file.name, progress=progress)
                
                # Extract text
                text = self.extract_text_from_pdf(pdf_file)
                if not text:
                    continue
                
                # Create chunks
                chunks = self.chunk_text(text, pdf_file.name)
                all_chunks.extend(chunks)
            
            if not all_chunks:
                self.update_status('error', 'No text could be extracted from PDF files')
                return {
                    'success': False,
                    'message': 'No text extracted from PDF files',
                    'files_processed': len(pdf_files)
                }
            
            # Add chunks to ChromaDB
            self.update_status('processing', 'Adding chunks to vector database...', progress=85)
            
            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                
                ids = [chunk['id'] for chunk in batch]
                documents = [chunk['text'] for chunk in batch]
                metadatas = [chunk['metadata'] for chunk in batch]
                
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            # Save metadata
            self.save_processing_metadata(all_chunks, len(pdf_files))
            
            self.update_status('completed', 
                             f'Successfully processed {len(pdf_files)} files, {len(all_chunks)} chunks created',
                             progress=100)
            
            return {
                'success': True,
                'message': 'PDF processing completed successfully',
                'files_processed': len(pdf_files),
                'total_chunks': len(all_chunks),
                'collection_info': {
                    'name': 'rag_documents',
                    'count': len(all_chunks)
                }
            }
            
        except Exception as e:
            error_msg = f'Error during PDF processing: {str(e)}'
            self.update_status('error', error_msg)
            return {
                'success': False,
                'message': error_msg,
                'files_processed': 0
            }
    
    def save_processing_metadata(self, chunks: List[Dict], files_count: int):
        """Save metadata about the processing"""
        metadata = {
            'processing_timestamp': datetime.now().isoformat(),
            'config': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'embedding_model': self.embedding_model,
            },
            'statistics': {
                'total_chunks': len(chunks),
                'total_characters': sum(chunk['metadata']['char_count'] for chunk in chunks),
                'total_words': sum(chunk['metadata']['word_count'] for chunk in chunks),
                'avg_chunk_size': sum(chunk['metadata']['char_count'] for chunk in chunks) / len(chunks) if chunks else 0,
                'files_processed': files_count,
                'collection_info': {
                    'name': 'rag_documents',
                    'count': len(chunks)
                }
            }
        }
        
        metadata_path = self.base_dir / "processing_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Processing metadata saved to {metadata_path}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get current ChromaDB collection statistics"""
        try:
            chroma_client = chromadb.PersistentClient(path="/app/chromadb_data")
            
            try:
                collection = chroma_client.get_collection("rag_documents")
                count = collection.count()
                collection_info = {
                    'name': 'rag_documents',
                    'count': count
                }
            except:
                collection_info = {
                    'name': 'N/A',
                    'count': 0
                }
            
            # Try to load processing metadata
            metadata_path = self.base_dir / "processing_metadata.json"
            processing_metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    processing_metadata = json.load(f)
            
            return {
                'collection_info': collection_info,
                'processing_metadata': processing_metadata,
                'pdf_files': [f.name for f in self.get_pdf_files()],
                'last_processing': processing_metadata.get('processing_timestamp', 'Never')
            }
        except Exception as e:
            return {
                'error': str(e),
                'collection_info': {'name': 'Error', 'count': 0},
                'processing_metadata': {},
                'pdf_files': [f.name for f in self.get_pdf_files()],
                'last_processing': 'Error retrieving information'
            }