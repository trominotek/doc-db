#!/usr/bin/env python3
"""
Simple PDF Uploader with ChromaDB Built-in Embeddings
Reads PDFs, creates intelligent chunks, and stores in ChromaDB using built-in embeddings
"""

import os
import sys
import re
import uuid
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import argparse
import logging

try:
    import PyPDF2
    PDF_READER = "PyPDF2"
except ImportError:
    try:
        import pdfplumber
        PDF_READER = "pdfplumber"
    except ImportError:
        PDF_READER = None

import chromadb
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplePDFUploader:
    """Simple PDF processing with ChromaDB built-in embeddings"""
    
    def __init__(self, 
                 chromadb_path: str = "/Users/tojojose/trominos/doc-db",
                 collection_name: Optional[str] = None):
        """
        Initialize Simple PDF Uploader
        
        Args:
            chromadb_path: Path to ChromaDB storage
            collection_name: ChromaDB collection name (auto-generated if None)
        """
        self.chromadb_path = chromadb_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        logger.info(f"Initializing ChromaDB at {chromadb_path}")
        self.chroma_client = chromadb.PersistentClient(path=chromadb_path)
        
        # Download NLTK data if available
        if NLTK_AVAILABLE:
            self._ensure_nltk_data()
        
        # Processing stats
        self.stats = {
            'pages_processed': 0,
            'chunks_created': 0,
            'chunks_stored': 0,
            'errors': []
        }

    def _ensure_nltk_data(self):
        """Download required NLTK data if not present"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page-level metadata
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries with page text and metadata
        """
        if not PDF_READER:
            raise ImportError("No PDF reading library available. Install PyPDF2 or pdfplumber")
        
        logger.info(f"Extracting text from PDF: {pdf_path}")
        pages_data = []
        
        if PDF_READER == "PyPDF2":
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():  # Only include pages with content
                            pages_data.append({
                                'page_number': page_num + 1,
                                'text': text,
                                'char_count': len(text),
                                'word_count': len(text.split())
                            })
                            self.stats['pages_processed'] += 1
                    except Exception as e:
                        error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                        logger.warning(error_msg)
                        self.stats['errors'].append(error_msg)
        
        elif PDF_READER == "pdfplumber":
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            pages_data.append({
                                'page_number': page_num + 1,
                                'text': text,
                                'char_count': len(text),
                                'word_count': len(text.split())
                            })
                            self.stats['pages_processed'] += 1
                    except Exception as e:
                        error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                        logger.warning(error_msg)
                        self.stats['errors'].append(error_msg)
        
        logger.info(f"Extracted text from {len(pages_data)} pages")
        return pages_data

    def simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting without NLTK"""
        # Split on periods, exclamation marks, and question marks
        sentences = re.split(r'[.!?]+', text)
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def chunk_text(self, 
                   pages_data: List[Dict[str, Any]], 
                   chunk_size: int = 1000, 
                   overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Create text chunks with context preservation
        
        Args:
            pages_data: List of page data from extract_text_from_pdf
            chunk_size: Target size for each chunk (characters)
            overlap: Overlap between chunks (characters)
            
        Returns:
            List of chunk dictionaries with metadata
        """
        logger.info(f"Creating chunks (size: {chunk_size}, overlap: {overlap})")
        chunks = []
        
        for page_data in pages_data:
            page_text = page_data['text']
            page_num = page_data['page_number']
            
            # Split into sentences
            if NLTK_AVAILABLE:
                sentences = sent_tokenize(page_text)
            else:
                sentences = self.simple_sentence_split(page_text)
            
            current_chunk = ""
            current_chunk_sentences = []
            
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(potential_chunk) <= chunk_size:
                    current_chunk = potential_chunk
                    current_chunk_sentences.append(sentence)
                else:
                    # Save current chunk if it has content
                    if current_chunk.strip():
                        chunk_data = self._create_chunk_metadata(
                            current_chunk.strip(), 
                            page_num, 
                            current_chunk_sentences,
                            len(chunks)
                        )
                        chunks.append(chunk_data)
                        self.stats['chunks_created'] += 1
                    
                    # Start new chunk with overlap
                    if overlap > 0 and len(current_chunk_sentences) > 1:
                        # Take last few sentences for overlap
                        overlap_text = ""
                        overlap_sentences = []
                        
                        for i in range(len(current_chunk_sentences) - 1, -1, -1):
                            test_overlap = current_chunk_sentences[i] + " " + overlap_text if overlap_text else current_chunk_sentences[i]
                            if len(test_overlap) <= overlap:
                                overlap_text = test_overlap
                                overlap_sentences.insert(0, current_chunk_sentences[i])
                            else:
                                break
                        
                        current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                        current_chunk_sentences = overlap_sentences + [sentence]
                    else:
                        current_chunk = sentence
                        current_chunk_sentences = [sentence]
            
            # Don't forget the last chunk
            if current_chunk.strip():
                chunk_data = self._create_chunk_metadata(
                    current_chunk.strip(), 
                    page_num, 
                    current_chunk_sentences,
                    len(chunks)
                )
                chunks.append(chunk_data)
                self.stats['chunks_created'] += 1
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def _create_chunk_metadata(self, 
                              text: str, 
                              page_num: int, 
                              sentences: List[str],
                              chunk_id: int) -> Dict[str, Any]:
        """Create comprehensive metadata for a chunk"""
        
        # Generate unique ID
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        unique_id = f"chunk_{chunk_id:04d}_{page_num:03d}_{text_hash}"
        
        # Extract keywords (simple approach)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords = [word for word, freq in top_keywords]
        
        return {
            'id': unique_id,
            'text': text,
            'page_number': page_num,
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sentences),
            'keywords': keywords,
            'created_at': datetime.now().isoformat(),
            'chunk_index': chunk_id
        }

    def store_in_chromadb(self, 
                         chunks: List[Dict[str, Any]], 
                         pdf_filename: str,
                         collection_name: Optional[str] = None) -> str:
        """
        Store chunks in ChromaDB with built-in embeddings
        
        Args:
            chunks: List of chunks
            pdf_filename: Name of source PDF file
            collection_name: Override collection name
            
        Returns:
            Collection name used
        """
        # Determine collection name
        if collection_name:
            coll_name = collection_name
        elif self.collection_name:
            coll_name = self.collection_name
        else:
            # Generate collection name from PDF filename
            base_name = Path(pdf_filename).stem.lower()
            # Clean filename for collection name
            coll_name = re.sub(r'[^a-z0-9_]', '_', base_name)[:50]
        
        logger.info(f"Storing {len(chunks)} chunks in collection: {coll_name}")
        
        # Create or get collection with built-in embeddings
        try:
            collection = self.chroma_client.create_collection(
                name=coll_name,
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "source_pdf": pdf_filename,
                    "total_chunks": len(chunks),
                    "embedding_model": "chromadb_builtin",
                    "uploader_version": "simple_1.0"
                }
            )
            logger.info(f"Created new collection: {coll_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Using existing collection: {coll_name}")
                collection = self.chroma_client.get_collection(name=coll_name)
            else:
                raise e
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk['id'])
            documents.append(chunk['text'])
            
            # Prepare metadata
            metadata = {
                'page_number': chunk['page_number'],
                'char_count': chunk['char_count'],
                'word_count': chunk['word_count'],
                'sentence_count': chunk['sentence_count'],
                'chunk_index': chunk['chunk_index'],
                'created_at': chunk['created_at'],
                'source_pdf': pdf_filename,
                'keywords': ','.join(chunk['keywords'][:5])  # Store top 5 keywords as string
            }
            metadatas.append(metadata)
        
        # Store in ChromaDB in batches (ChromaDB will handle embeddings automatically)
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            logger.info(f"Storing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            try:
                collection.add(
                    ids=ids[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                self.stats['chunks_stored'] += (end_idx - i)
            except Exception as e:
                error_msg = f"Error storing batch {i//batch_size + 1}: {str(e)}"
                logger.error(error_msg)
                self.stats['errors'].append(error_msg)
        
        logger.info(f"Successfully stored {self.stats['chunks_stored']} chunks in ChromaDB")
        return coll_name

    def process_pdf(self, 
                   pdf_path: str, 
                   chunk_size: int = 1000, 
                   overlap: int = 200,
                   collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete PDF processing pipeline
        
        Args:
            pdf_path: Path to PDF file
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            collection_name: ChromaDB collection name
            
        Returns:
            Processing results and statistics
        """
        logger.info(f"Starting PDF processing pipeline for: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Reset stats
        self.stats = {
            'pages_processed': 0,
            'chunks_created': 0,
            'chunks_stored': 0,
            'errors': []
        }
        
        try:
            # Step 1: Extract text from PDF
            pages_data = self.extract_text_from_pdf(pdf_path)
            
            # Step 2: Create chunks
            chunks = self.chunk_text(pages_data, chunk_size, overlap)
            
            # Step 3: Store in ChromaDB
            pdf_filename = os.path.basename(pdf_path)
            collection_used = self.store_in_chromadb(chunks, pdf_filename, collection_name)
            
            # Compile results
            results = {
                'success': True,
                'pdf_path': pdf_path,
                'collection_name': collection_used,
                'statistics': self.stats.copy(),
                'processing_time': datetime.now().isoformat()
            }
            
            logger.info("PDF processing completed successfully!")
            logger.info(f"Statistics: {self.stats}")
            
            return results
            
        except Exception as e:
            error_msg = f"PDF processing failed: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'statistics': self.stats.copy()
            }

def main():
    """Command line interface for simple PDF uploader"""
    parser = argparse.ArgumentParser(description='Upload PDF to ChromaDB with built-in embeddings')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--chromadb-path', default='/Users/tojojose/trominos/doc-db',
                       help='Path to ChromaDB storage directory')
    parser.add_argument('--collection-name', help='ChromaDB collection name')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Target chunk size in characters')
    parser.add_argument('--overlap', type=int, default=200,
                       help='Overlap between chunks in characters')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize uploader
    uploader = SimplePDFUploader(
        chromadb_path=args.chromadb_path,
        collection_name=args.collection_name
    )
    
    # Process PDF
    results = uploader.process_pdf(
        pdf_path=args.pdf_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        collection_name=args.collection_name
    )
    
    # Print results
    if results['success']:
        print(f"\n✅ PDF processing completed successfully!")
        print(f"Collection: {results['collection_name']}")
        print(f"Pages processed: {results['statistics']['pages_processed']}")
        print(f"Chunks created: {results['statistics']['chunks_created']}")
        print(f"Chunks stored: {results['statistics']['chunks_stored']}")
        
        if results['statistics']['errors']:
            print(f"\n⚠️  Warnings/Errors: {len(results['statistics']['errors'])}")
            for error in results['statistics']['errors']:
                print(f"  - {error}")
    else:
        print(f"\n❌ PDF processing failed!")
        print(f"Error: {results['error']}")
        if results['statistics']['errors']:
            print("Additional errors:")
            for error in results['statistics']['errors']:
                print(f"  - {error}")

if __name__ == "__main__":
    main()