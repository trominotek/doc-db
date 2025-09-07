#!/usr/bin/env python3
"""
PDF Processing Service for Doc-DB
Processes PDFs from pdfs/ folder, chunks them, creates embeddings, and updates ChromaDB
"""

import os
import sys
import json
import uuid
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Import the RAG-quiz components
rag_quiz_path = Path(__file__).parent.parent / "rag-quiz" / "src"
sys.path.append(str(rag_quiz_path))

try:
    from step1_config import RAGConfig
    from step2_layout_extract import LayoutExtractor
    from step3_sectioner import DocumentSectioner
    from step4_chunker import TextChunker, Chunk
    from step5_build_and_ingest import ChromaDBManager, RAGPipelineBuilder
except ImportError as e:
    print(f"Error importing RAG-quiz components: {e}")
    print(f"Make sure RAG-quiz is properly set up at: {rag_quiz_path}")
    sys.exit(1)

class DocDBConfig(RAGConfig):
    """Extended config for doc-db PDF processing"""
    
    def __init__(self):
        super().__init__()
        # Override paths for doc-db
        self.BASE_DIR = Path(__file__).parent
        self.PDF_DIR = self.BASE_DIR / "pdfs"
        self.CHROMA_DIR = self.BASE_DIR
        self.DATA_DIR = self.BASE_DIR / "processed_data"
        
        # Create directories
        self.PDF_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
        (self.DATA_DIR / "extracted").mkdir(exist_ok=True)
        (self.DATA_DIR / "sections").mkdir(exist_ok=True)
        (self.DATA_DIR / "chunks").mkdir(exist_ok=True)

class DocumentProcessor:
    """Process PDFs for the doc-db system"""
    
    def __init__(self):
        self.config = DocDBConfig()
        self.rag_builder = RAGPipelineBuilder(self.config)
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
        pdf_files = list(self.config.PDF_DIR.glob("*.pdf"))
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
            
            # Step 1: Extract content from all PDFs
            self.update_status('processing', 'Extracting content from PDFs...', progress=10)
            extractor = LayoutExtractor(self.config)
            extracted_content = {}
            
            for i, pdf_file in enumerate(pdf_files):
                self.update_status('processing', f'Extracting: {pdf_file.name}', 
                                 current_file=pdf_file.name, progress=10 + (i * 20 // len(pdf_files)))
                content = extractor.extract_from_pdf(pdf_file)
                if content:
                    extracted_content[pdf_file.name] = content
            
            if not extracted_content:
                self.update_status('error', 'Failed to extract content from any PDF files')
                return {
                    'success': False,
                    'message': 'Failed to extract content from PDF files',
                    'files_processed': 0
                }
            
            # Step 2: Section documents
            self.update_status('processing', 'Sectioning documents...', progress=30)
            sectioner = DocumentSectioner(self.config)
            all_sections = sectioner.process_all_documents(extracted_content)
            
            # Step 3: Chunk sections
            self.update_status('processing', 'Chunking sections...', progress=50)
            chunker = TextChunker(self.config)
            chunks = chunker.chunk_documents(all_sections)
            optimized_chunks = chunker.optimize_chunks(chunks)
            
            # Step 4: Create ChromaDB collection
            self.update_status('processing', 'Creating vector database...', progress=70)
            if not self.rag_builder.chroma_manager.create_collection("rag_documents", reset=reset_collection):
                self.update_status('error', 'Failed to create ChromaDB collection')
                return {
                    'success': False,
                    'message': 'Failed to create ChromaDB collection',
                    'files_processed': len(extracted_content)
                }
            
            # Step 5: Add chunks to database
            self.update_status('processing', 'Adding chunks to vector database...', progress=85)
            if not self.rag_builder.chroma_manager.add_chunks(optimized_chunks):
                self.update_status('error', 'Failed to add chunks to ChromaDB')
                return {
                    'success': False,
                    'message': 'Failed to add chunks to ChromaDB',
                    'files_processed': len(extracted_content)
                }
            
            # Step 6: Save metadata
            collection_info = self.rag_builder.chroma_manager.get_collection_info()
            self.save_processing_metadata(optimized_chunks, collection_info, list(extracted_content.keys()))
            
            self.update_status('completed', 
                             f'Successfully processed {len(extracted_content)} files, {len(optimized_chunks)} chunks created',
                             progress=100)
            
            return {
                'success': True,
                'message': 'PDF processing completed successfully',
                'files_processed': len(extracted_content),
                'total_chunks': len(optimized_chunks),
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
    
    def save_processing_metadata(self, chunks: List[Chunk], collection_info: Dict[str, Any], files_processed: List[str]):
        """Save metadata about the processing"""
        metadata = {
            'processing_timestamp': datetime.now().isoformat(),
            'config': {
                'chunk_size': self.config.CHUNK_SIZE,
                'chunk_overlap': self.config.CHUNK_OVERLAP,
                'embedding_model': self.config.EMBEDDING_MODEL,
            },
            'statistics': {
                'total_chunks': len(chunks),
                'total_characters': sum(chunk.char_count for chunk in chunks),
                'total_words': sum(chunk.word_count for chunk in chunks),
                'avg_chunk_size': sum(chunk.char_count for chunk in chunks) / len(chunks) if chunks else 0,
                'files_processed': files_processed,
                'collection_info': collection_info
            }
        }
        
        metadata_path = self.config.BASE_DIR / "processing_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Processing metadata saved to {metadata_path}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get current ChromaDB collection statistics"""
        try:
            collection_info = self.rag_builder.chroma_manager.get_collection_info()
            
            # Try to load processing metadata
            metadata_path = self.config.BASE_DIR / "processing_metadata.json"
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
                'collection_info': {},
                'processing_metadata': {},
                'pdf_files': [f.name for f in self.get_pdf_files()],
                'last_processing': 'Error retrieving information'
            }

def main():
    """Command line interface"""
    processor = DocumentProcessor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'process':
            print("Starting PDF processing...")
            result = processor.process_all_pdfs(reset_collection=True)
            
            if result['success']:
                print(f"\n✅ Processing completed successfully!")
                print(f"   Files processed: {result['files_processed']}")
                print(f"   Total chunks: {result['total_chunks']}")
                print(f"   Collection: {result['collection_info'].get('name', 'N/A')}")
            else:
                print(f"\n❌ Processing failed: {result['message']}")
                
        elif command == 'status':
            stats = processor.get_collection_stats()
            print(f"\n📊 Collection Statistics:")
            print(f"   Collection name: {stats['collection_info'].get('name', 'N/A')}")
            print(f"   Total chunks: {stats['collection_info'].get('count', 0)}")
            print(f"   PDF files found: {len(stats['pdf_files'])}")
            print(f"   Last processing: {stats['last_processing']}")
            
            if stats['pdf_files']:
                print(f"\n📄 PDF Files:")
                for pdf_file in stats['pdf_files']:
                    print(f"   • {pdf_file}")
                    
        else:
            print("Usage: python pdf_processor.py [process|status]")
    else:
        print("Usage: python pdf_processor.py [process|status]")
        print("  process - Process all PDFs in pdfs/ folder")
        print("  status  - Show collection statistics")

if __name__ == "__main__":
    main()