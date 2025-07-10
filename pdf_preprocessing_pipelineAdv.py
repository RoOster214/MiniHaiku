#!/usr/bin/env python3
"""
pdf_preprocessing_pipelineAdv.py
Simple but effective PDF processor for AI training data.
Maximizes content extraction while keeping text clean and readable.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict
import PyPDF2
import pdfplumber
from nltk.tokenize import sent_tokenize
import nltk

# Download punkt tokenizer if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplePDFProcessor:
    """Simple, effective PDF processor for AI training"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Simple, permissive settings
        self.min_words = 10
        self.max_words = 1000
        self.min_chunk_quality = 0.3  # Very permissive
        
    def process_folder(self, pdf_dir: str, subject: str):
        """Process all PDFs in a folder"""
        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            logger.error(f"Directory not found: {pdf_path}")
            return
            
        pdf_files = list(pdf_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {subject}")
        
        successful = 0
        total_chunks = 0
        
        for pdf_file in pdf_files:
            # Quick file validation
            if not self._is_valid_pdf(pdf_file):
                logger.debug(f"Skipping invalid PDF: {pdf_file.name}")
                continue
                
            try:
                chunks = self._process_single_pdf(pdf_file, subject)
                if chunks:
                    # Save chunks
                    output_file = self.output_dir / subject / f"{pdf_file.stem}.jsonl"
                    output_file.parent.mkdir(exist_ok=True, parents=True)
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for chunk in chunks:
                            json.dump(chunk, f, ensure_ascii=False)
                            f.write('\n')
                    
                    successful += 1
                    total_chunks += len(chunks)
                    logger.info(f"✓ {pdf_file.name}: {len(chunks)} chunks")
                else:
                    logger.debug(f"✗ {pdf_file.name}: No chunks extracted")
                    
            except Exception as e:
                logger.debug(f"✗ {pdf_file.name}: {str(e)}")
                
        logger.info(f"\n{subject} Summary:")
        logger.info(f"  Processed: {successful}/{len(pdf_files)} files")
        logger.info(f"  Chunks: {total_chunks}")
        logger.info(f"  Avg chunks/file: {total_chunks/max(successful, 1):.1f}")
        
    def _is_valid_pdf(self, pdf_path: Path) -> bool:
        """Quick PDF validation"""
        try:
            # Check file size
            if pdf_path.stat().st_size < 1024:  # Less than 1KB
                return False
                
            # Check PDF header
            with open(pdf_path, 'rb') as f:
                header = f.read(8)
                return header.startswith(b'%PDF-')
        except:
            return False
            
    def _process_single_pdf(self, pdf_path: Path, subject: str) -> List[Dict]:
        """Process a single PDF file"""
        # Extract text
        text = self._extract_text(pdf_path)
        if not text or len(text.split()) < 50:
            return []
            
        # Clean text
        text = self._clean_text(text)
        
        # Create chunks
        chunks = self._create_chunks(text, subject, pdf_path.name)
        
        return chunks
        
    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text using multiple methods"""
        text = ""
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i > 100:  # Limit pages for very long documents
                        break
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        
            if text and len(text.split()) > 50:
                return text
        except:
            pass
            
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(reader.pages):
                    if i > 100:  # Limit pages
                        break
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        
            return text
        except:
            return ""
            
    def _clean_text(self, text: str) -> str:
        """Simple but effective text cleaning"""
        # Remove common LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)  # \command{text} -> text
        text = re.sub(r'\\[a-zA-Z]+\*?\s*', ' ', text)  # \command -> space
        
        # Clean math delimiters but keep content
        text = re.sub(r'\$\$([^$]+)\$\$', r' \1 ', text)
        text = re.sub(r'\$([^$]+)\$', r' \1 ', text)
        
        # Remove extra brackets and slashes
        text = re.sub(r'[{}\\]', ' ', text)
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
        
    def _create_chunks(self, text: str, subject: str, filename: str) -> List[Dict]:
        """Create simple chunks from text"""
        chunks = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Also split very long paragraphs by sentences
        all_segments = []
        for para in paragraphs:
            words = para.split()
            if len(words) > self.max_words:
                # Split by sentences
                sentences = sent_tokenize(para)
                current = ""
                for sent in sentences:
                    if len((current + " " + sent).split()) <= self.max_words:
                        current = current + " " + sent if current else sent
                    else:
                        if current and len(current.split()) >= self.min_words:
                            all_segments.append(current)
                        current = sent
                if current and len(current.split()) >= self.min_words:
                    all_segments.append(current)
            elif len(words) >= self.min_words:
                all_segments.append(para)
                
        # Process segments into chunks
        for segment in all_segments:
            if self._is_good_chunk(segment):
                chunk = {
                    'text': segment.strip(),
                    'subject': subject,
                    'source': filename,
                    'word_count': len(segment.split())
                }
                chunks.append(chunk)
                
        return chunks
        
    def _is_good_chunk(self, text: str) -> bool:
        """Simple quality check"""
        words = text.split()
        
        # Length check
        if len(words) < self.min_words or len(words) > self.max_words:
            return False
            
        # Must have some letters (not just numbers/symbols)
        if not re.search(r'[a-zA-Z]{3,}', text):
            return False
            
        # Must have at least one sentence-like structure
        if not re.search(r'[.!?]\s*[A-Z]|[.!?]$', text):
            # Allow chunks without periods if they have other good signals
            if not any(word in text.lower() for word in 
                      ['the', 'is', 'are', 'was', 'were', 'and', 'or', 'but']):
                return False
                
        return True


def main():
    """Process all PDFs for AI training"""
    # Hardcoded paths
    input_dir = 'E:/AI/Books'
    output_dir = 'E:/AI/Training_Data/processed_books'
    
    # Subject folders to process
    subjects = {
        'Math': 'mathematics',
        'Algebra': 'algebra',
        'Logic': 'logic',
        'Science': 'science'
    }
    
    processor = SimplePDFProcessor(output_dir)
    
    total_stats = {
        'files_found': 0,
        'files_processed': 0,
        'chunks_created': 0
    }
    
    print(f"\n{'='*60}")
    print(f"PDF Processing for AI Training")
    print(f"{'='*60}\n")
    
    for folder, subject in subjects.items():
        folder_path = Path(input_dir) / folder
        if folder_path.exists():
            print(f"\nProcessing {folder} → {subject}...")
            print(f"{'-'*40}")
            
            # Count files before
            pdf_count = len(list(folder_path.glob("*.pdf")))
            total_stats['files_found'] += pdf_count
            
            # Process
            processor.process_folder(str(folder_path), subject)
            
            # Count results
            output_path = Path(output_dir) / subject
            if output_path.exists():
                processed = len(list(output_path.glob("*.jsonl")))
                total_stats['files_processed'] += processed
                
                # Count chunks
                chunk_count = 0
                for jsonl_file in output_path.glob("*.jsonl"):
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        chunk_count += sum(1 for _ in f)
                total_stats['chunks_created'] += chunk_count
        else:
            print(f"\n⚠️  Folder not found: {folder_path}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"📁 Total PDFs found: {total_stats['files_found']}")
    print(f"✅ Successfully processed: {total_stats['files_processed']}")
    print(f"📄 Total chunks created: {total_stats['chunks_created']}")
    
    if total_stats['files_processed'] > 0:
        avg = total_stats['chunks_created'] / total_stats['files_processed']
        print(f"📊 Average chunks per file: {avg:.1f}")
    
    print(f"\n💾 Output saved to: {output_dir}")
    
    # Quality assessment
    if total_stats['chunks_created'] >= 5000:
        print(f"\n🎉 EXCELLENT! {total_stats['chunks_created']} chunks is great for training!")
    elif total_stats['chunks_created'] >= 1000:
        print(f"\n👍 GOOD! {total_stats['chunks_created']} chunks should work for training.")
    else:
        print(f"\n⚠️  Only {total_stats['chunks_created']} chunks. Consider adding more PDFs.")


if __name__ == "__main__":
    main()