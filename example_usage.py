#!/usr/bin/env python3
"""
Example usage of the comprehensive PDF uploader
"""

from pdf_uploader import PDFUploader
import os

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic PDF Upload Example ===")
    
    # Initialize uploader
    uploader = PDFUploader(
        chromadb_path="/Users/tojojose/trominos/doc-db",
        embedding_model="all-MiniLM-L6-v2"  # Fast, good quality model
    )
    
    # Check if PDF exists
    pdf_path = "/Users/tojojose/trominos/doc-db/00_afh_full.pdf"
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return
    
    # Process PDF
    results = uploader.process_pdf(
        pdf_path=pdf_path,
        chunk_size=1000,           # 1000 characters per chunk
        overlap=200,               # 200 character overlap
        collection_name="aviation_handbook_v2"
    )
    
    # Print results
    if results['success']:
        print("✅ Upload successful!")
        print(f"Collection: {results['collection_name']}")
        print(f"Pages: {results['statistics']['pages_processed']}")
        print(f"Chunks: {results['statistics']['chunks_created']}")
        print(f"Stored: {results['statistics']['chunks_stored']}")
    else:
        print("❌ Upload failed!")
        print(f"Error: {results['error']}")

def example_custom_settings():
    """Advanced usage with custom settings"""
    print("=== Advanced PDF Upload Example ===")
    
    # Initialize with different embedding model
    uploader = PDFUploader(
        chromadb_path="/Users/tojojose/trominos/doc-db",
        embedding_model="all-mpnet-base-v2"  # Higher quality, slower model
    )
    
    pdf_path = "/Users/tojojose/trominos/doc-db/00_afh_full.pdf"
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return
    
    # Process with custom chunk settings
    results = uploader.process_pdf(
        pdf_path=pdf_path,
        chunk_size=1500,           # Larger chunks
        overlap=300,               # More overlap for better context
        collection_name="aviation_handbook_detailed"
    )
    
    print("Results:", results)

def example_multiple_pdfs():
    """Process multiple PDFs into different collections"""
    print("=== Multiple PDF Processing Example ===")
    
    uploader = PDFUploader()
    
    # List of PDFs to process (add your own paths)
    pdfs_to_process = [
        {
            "path": "/Users/tojojose/trominos/doc-db/00_afh_full.pdf",
            "collection": "aviation_handbook",
            "chunk_size": 1000
        }
        # Add more PDFs here
    ]
    
    for pdf_config in pdfs_to_process:
        if os.path.exists(pdf_config["path"]):
            print(f"\nProcessing: {pdf_config['path']}")
            results = uploader.process_pdf(
                pdf_path=pdf_config["path"],
                chunk_size=pdf_config.get("chunk_size", 1000),
                collection_name=pdf_config["collection"]
            )
            
            if results['success']:
                print(f"✅ {pdf_config['collection']}: {results['statistics']['chunks_stored']} chunks")
            else:
                print(f"❌ Failed: {results['error']}")

if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    print("\n" + "="*50 + "\n")
    # example_custom_settings()  # Uncomment to run
    # example_multiple_pdfs()    # Uncomment to run