# improved_general80_literature_training.py
"""
Enhanced Educational AI Training System
Combines General80 explanatory excellence with Literature engagement
With proper data balancing, curriculum learning, and advanced training techniques
WITH AUTO-RESUME FROM LATEST CHECKPOINT - FIXED VERSION
NOW SUPPORTS BOTH .TXT AND .JSONL FILES IN GENERAL80 DIRECTORY
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast, GradScaler
import json
import time
import random
import re
import logging
import os
import math
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
from spm_tokenizer import HaikuSPMTokenizer

try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
from tqdm import tqdm

# Import your model
from validation_config import ValidationConfig
from validation_model import create_validation_model

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

def default_metrics_dict():
    return {'loss': 0.0, 'count': 0}

class WordLevelTokenizer:
    """Improved word-level tokenizer for better learning efficiency"""
    
    def __init__(self, vocab_size: int = 32000, min_freq: int = 3, reasoning_tokens: Dict[str, int] = None):

        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.reasoning_tokens = reasoning_tokens or {}
        self.word_to_id = {}
        self.id_to_word = {}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
    def build_vocab(self, documents: List[Dict]):
        """Build vocabulary from documents"""
        logger.info("Building vocabulary from corpus...")
        
        # Count word frequencies
        word_freq = Counter()
        
        for doc in tqdm(documents, desc="Counting words"):
            text = doc['text'].lower()
            # Simple tokenization - can be improved with proper tokenizer
            words = re.findall(r'\b\w+\b|[^\w\s]', text)
            word_freq.update(words)
        
        # Initialize with special tokens
        self.word_to_id = {
            self.pad_token: self.pad_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.unk_token: self.unk_token_id
        }
        # Reserve space for reasoning tokens if provided
        base_vocab_size = self.vocab_size
        if self.reasoning_tokens:
            base_vocab_size = self.vocab_size - len(self.reasoning_tokens)

        # Add frequent words
        vocab_candidates = [(w, f) for w, f in word_freq.items() if f >= self.min_freq]
        vocab_candidates.sort(key=lambda x: x[1], reverse=True)
        
        for word, freq in vocab_candidates[:self.vocab_size - 4]:
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)

        # Add reasoning tokens
        if self.reasoning_tokens:
            for token_str, token_id in self.reasoning_tokens.items():
                 self.word_to_id[token_str] = token_id

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        logger.info(f"Vocabulary size: {len(self.word_to_id)}")
        
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs"""
        text = text.lower()
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        tokens = [self.bos_token_id]
        for word in words:
            token_id = self.word_to_id.get(word, self.unk_token_id)
            tokens.append(token_id)
        tokens.append(self.eos_token_id)
        
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.eos_token_id]
            
        return tokens
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text"""
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                word = self.id_to_word[token_id]
                if skip_special and word in [self.pad_token, self.bos_token, 
                                            self.eos_token, self.unk_token]:
                    continue
                words.append(word)
        
        # Simple detokenization
        text = ' '.join(words)
        # Fix punctuation spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'(["\'])\s+', r'\1', text)
        text = re.sub(r'\s+(["\'])', r' \1', text)
        
        return text
    
    def save(self, path: str):
        """Save tokenizer to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word_to_id': self.word_to_id,
                'id_to_word': self.id_to_word,
                'vocab_size': self.vocab_size,
                'min_freq': self.min_freq
            }, f)
    
    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word_to_id = data['word_to_id']
            self.id_to_word = data['id_to_word']
            self.vocab_size = data['vocab_size']
            self.min_freq = data['min_freq']

class BalancedBatchSampler(Sampler):
    """Custom sampler to ensure balanced batches"""
    
    def __init__(self, dataset, batch_size: int, general80_ratio: float = 0.7):
        self.dataset = dataset
        self.batch_size = batch_size
        self.general80_ratio = general80_ratio
        
        # Get indices for each content type
        self.general80_indices = []
        self.literature_indices = []
        
        for idx, doc in enumerate(dataset.documents):
            if doc['content_type'] == 'explanatory':
                self.general80_indices.append(idx)
            else:
                self.literature_indices.append(idx)
        
        # Calculate samples per batch
        self.general80_per_batch = int(batch_size * general80_ratio)
        self.literature_per_batch = batch_size - self.general80_per_batch
        
        # Calculate total batches
        self.num_batches = min(
            len(self.general80_indices) // max(self.general80_per_batch, 1),
            len(self.literature_indices) // max(self.literature_per_batch, 1)
        )
        
        logger.info(f"Balanced sampler: {self.general80_per_batch} explanatory + "
                   f"{self.literature_per_batch} literature per batch")
    
    def __iter__(self):
        # Shuffle indices
        random.shuffle(self.general80_indices)
        random.shuffle(self.literature_indices)
        
        for batch_idx in range(self.num_batches):
            batch_indices = []
            
            # Add general80 samples
            start_idx = batch_idx * self.general80_per_batch
            end_idx = start_idx + self.general80_per_batch
            batch_indices.extend(self.general80_indices[start_idx:end_idx])
            
            # Add literature samples
            start_idx = batch_idx * self.literature_per_batch
            end_idx = start_idx + self.literature_per_batch
            batch_indices.extend(self.literature_indices[start_idx:end_idx])
            
            # Shuffle within batch
            random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        return self.num_batches

class EducationalDataset(Dataset):
    """Enhanced dataset with curriculum learning and data augmentation"""
    
    def __init__(self, general80_dir: str, literature_dir: str, 
                 tokenizer: WordLevelTokenizer,
                 max_seq_len: int = 768,
                 min_seq_len: int = 384,
                 general80_ratio: float = 0.7,
                 quality_threshold: float = 0.65,
                 augment_general80: bool = True):
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.current_max_len = min_seq_len  # For curriculum learning
        self.general80_ratio = general80_ratio
        self.quality_threshold = quality_threshold
        self.augment_general80 = augment_general80
        
        # Load documents
        self.documents = []
        self.general80_docs = []
        self.literature_docs = []
        
        # Load corpora
        self._load_general80(general80_dir)
        self._load_literature(literature_dir)
        
        # Create balanced dataset
        self._create_balanced_dataset()
        
        # Build tokenizer vocabulary
        #self.tokenizer.build_vocab(self.documents)
        
        # Analyze dataset
        self._analyze_dataset()
    
    def _load_general80(self, general80_dir: str):
        """Load General80 explanatory content from both .txt and .jsonl files"""
        general80_path = Path(general80_dir)
        
        if not general80_path.exists():
            logger.warning(f"General80 directory not found: {general80_dir}")
            return
        
        logger.info(f"Loading General80 corpus from: {general80_path}")
        
        # Load .txt files
        txt_files = list(general80_path.rglob("*.txt"))
        logger.info(f"Found {len(txt_files)} .txt files in General80")
        
        for file_path in tqdm(txt_files, desc="Loading General80 .txt files"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                
                if len(content) < 100:  # Minimum content length
                    continue
                
                doc = {
                    'text': content,
                    'content_type': 'explanatory',
                    'source': 'general80',
                    'quality_score': 0.9,  # High quality by default
                    'file_path': str(file_path),
                    'file_type': 'txt'
                }
                
                self.general80_docs.append(doc)
                
                # Data augmentation for General80
                if self.augment_general80:
                    augmented = self._augment_explanatory(content)
                    for aug_text in augmented:
                        aug_doc = doc.copy()
                        aug_doc['text'] = aug_text
                        aug_doc['augmented'] = True
                        self.general80_docs.append(aug_doc)
                        
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        # Load .jsonl files  
        jsonl_files = list(general80_path.rglob("*.jsonl"))
        logger.info(f"Found {len(jsonl_files)} .jsonl files in General80")
        
        for file_path in tqdm(jsonl_files, desc="Loading General80 .jsonl files"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                content = data.get('text', '').strip()
                                
                                if len(content) < 100:  # Minimum content length
                                    continue
                                
                                doc = {
                                    'text': content,
                                    'content_type': 'explanatory',
                                    'source': 'general80', 
                                    'quality_score': data.get('quality_score', 0.9),  # Use provided score or default high
                                    'file_path': f"{file_path}:{line_num}",
                                    'file_type': 'jsonl'
                                }
                                
                                self.general80_docs.append(doc)
                                
                                # Data augmentation for General80
                                if self.augment_general80:
                                    augmented = self._augment_explanatory(content)
                                    for aug_text in augmented:
                                        aug_doc = doc.copy()
                                        aug_doc['text'] = aug_text
                                        aug_doc['augmented'] = True
                                        self.general80_docs.append(aug_doc)
                                        
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON decode error in {file_path}:{line_num}: {e}")
                                continue
                                
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.general80_docs)} General80 documents total")
        
        # Log breakdown by file type
        txt_count = sum(1 for doc in self.general80_docs if doc.get('file_type') == 'txt' and not doc.get('augmented', False))
        jsonl_count = sum(1 for doc in self.general80_docs if doc.get('file_type') == 'jsonl' and not doc.get('augmented', False))
        augmented_count = sum(1 for doc in self.general80_docs if doc.get('augmented', False))
        
        logger.info(f"  - From .txt files: {txt_count} documents")
        logger.info(f"  - From .jsonl files: {jsonl_count} documents") 
        logger.info(f"  - Augmented: {augmented_count} documents")
    
    def _augment_explanatory(self, text: str) -> List[str]:
        """Create variations of explanatory content"""
        augmented = []
        
        # 1. Alternative openings
        openings = {
            "To understand": ["Let's explore how", "We can see that", "The concept of"],
            "Let me explain": ["I'll describe how", "Here's how", "Allow me to show"],
            "The key": ["The main", "The essential", "The fundamental"]
        }
        
        for original, alternatives in openings.items():
            if text.startswith(original):
                for alt in alternatives[:1]:  # Use just one alternative
                    augmented.append(text.replace(original, alt, 1))
                break
        
        # 2. Add discourse markers
        sentences = text.split('. ')
        if len(sentences) > 3:
            markers = ["Furthermore, ", "Additionally, ", "Moreover, ", "In fact, "]
            marker_pos = len(sentences) // 2
            new_sentences = sentences[:marker_pos] + [random.choice(markers) + sentences[marker_pos].lower()] + sentences[marker_pos+1:]
            augmented.append('. '.join(new_sentences))
        
        return augmented[:2]  # Limit augmentation
    
    def _load_literature(self, literature_dir: str):
        """Load literature content with quality filtering"""
        lit_path = Path(literature_dir)
        
        if not lit_path.exists():
            logger.warning(f"Literature directory not found: {literature_dir}")
            return
        
        # Calculate target number based on General80 size
        target_lit_count = int(len(self.general80_docs) * (1 - self.general80_ratio) / self.general80_ratio)
        
        logger.info(f"Loading up to {target_lit_count} literature documents...")
        
        literature_files = list(lit_path.glob("literature_*.jsonl"))
        loaded_count = 0
        
        for file_path in literature_files:
            if loaded_count >= target_lit_count:
                break
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if loaded_count >= target_lit_count:
                            break
                            
                        try:
                            doc_data = json.loads(line.strip())
                            text = doc_data.get('text', '').strip()
                            quality = float(doc_data.get('quality_score', 0.5))
                            
                            # Quality filtering
                            if quality < self.quality_threshold:
                                continue
                            
                            # Length filtering
                            word_count = len(text.split())
                            if word_count < 100 or word_count > 2000:
                                continue
                            
                            # Basic coherence check
                            sentences = re.split(r'[.!?]+', text)
                            avg_sent_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
                            if avg_sent_len < 5 or avg_sent_len > 40:
                                continue
                            
                            doc = {
                                'text': text,
                                'content_type': 'literature',
                                'source': 'literature',
                                'quality_score': quality,
                                'file_path': str(file_path),
                                'file_type': 'jsonl'
                            }
                            
                            self.literature_docs.append(doc)
                            loaded_count += 1
                            
                        except (json.JSONDecodeError, ValueError):
                            continue
                            
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.literature_docs)} literature documents")
    
    def _create_balanced_dataset(self):
        """Create properly balanced dataset"""
        # Combine all documents
        self.documents = self.general80_docs + self.literature_docs
        
        # Shuffle
        random.shuffle(self.documents)
        
        # Create index mappings for balanced sampling
        self.content_type_indices = defaultdict(list)
        for idx, doc in enumerate(self.documents):
            self.content_type_indices[doc['content_type']].append(idx)
        
        logger.info(f"Total documents: {len(self.documents)}")
        for content_type, indices in self.content_type_indices.items():
            logger.info(f"  {content_type}: {len(indices)} documents")
    
    def _analyze_dataset(self):
        """Analyze dataset composition including file type breakdown"""
        stats = defaultdict(lambda: {'count': 0, 'total_words': 0, 'quality_sum': 0})
        file_type_stats = defaultdict(lambda: {'count': 0, 'augmented': 0})
        
        for doc in self.documents:
            content_type = doc['content_type']
            stats[content_type]['count'] += 1
            stats[content_type]['total_words'] += len(doc['text'].split())
            stats[content_type]['quality_sum'] += doc['quality_score']
            
            # Track file types
            file_type = doc.get('file_type', 'unknown')
            file_type_stats[file_type]['count'] += 1
            if doc.get('augmented', False):
                file_type_stats[file_type]['augmented'] += 1
        
        logger.info("\n📊 DATASET ANALYSIS:")
        logger.info("="*50)
        
        # Content type analysis
        for content_type, stat in stats.items():
            avg_words = stat['total_words'] / stat['count']
            avg_quality = stat['quality_sum'] / stat['count']
            logger.info(f"📋 {content_type.upper()}:")
            logger.info(f"    Documents: {stat['count']:,}")
            logger.info(f"    Avg words/doc: {avg_words:.0f}")
            logger.info(f"    Avg quality: {avg_quality:.3f}")
        
        # File type breakdown
        logger.info(f"\n📁 FILE TYPE BREAKDOWN:")
        for file_type, stat in file_type_stats.items():
            original_count = stat['count'] - stat['augmented']
            logger.info(f"    {file_type.upper()} files:")
            logger.info(f"      Original: {original_count:,}")
            logger.info(f"      Augmented: {stat['augmented']:,}")
            logger.info(f"      Total: {stat['count']:,}")
        
        # Overall statistics
        total_docs = len(self.documents)
        total_words = sum(stat['total_words'] for stat in stats.values())
        avg_doc_length = total_words / total_docs if total_docs > 0 else 0
        
        logger.info(f"\n📈 OVERALL STATISTICS:")
        logger.info(f"    Total documents: {total_docs:,}")
        logger.info(f"    Total words: {total_words:,}")
        logger.info(f"    Average doc length: {avg_doc_length:.0f} words")
        logger.info("="*50)
    
    def set_curriculum_length(self, epoch: int, max_epochs: int):
        """Update maximum sequence length for curriculum learning"""
        progress = min(epoch / max(max_epochs // 2, 1), 1.0)
        self.current_max_len = int(self.min_seq_len + 
                                  (self.max_seq_len - self.min_seq_len) * progress)
        logger.info(f"Curriculum learning: max sequence length = {self.current_max_len}")
    
    def create_train_val_split(self, val_ratio: float = 0.1):
        """Create training and validation datasets"""
        val_docs = []
        train_docs = []
        
        # Ensure balanced validation set
        for content_type, indices in self.content_type_indices.items():
            n_val = int(len(indices) * val_ratio)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            for idx in val_indices:
                val_docs.append(self.documents[idx])
            for idx in train_indices:
                train_docs.append(self.documents[idx])
        
        # Create new datasets
        train_dataset = EducationalDataset.__new__(EducationalDataset)
        val_dataset = EducationalDataset.__new__(EducationalDataset)
        
        # Copy attributes
        for attr in ['tokenizer', 'max_seq_len', 'min_seq_len', 'current_max_len']:
            setattr(train_dataset, attr, getattr(self, attr))
            setattr(val_dataset, attr, getattr(self, attr))
        
        train_dataset.documents = train_docs
        val_dataset.documents = val_docs
        
        # Rebuild indices
        for dataset in [train_dataset, val_dataset]:
            dataset.content_type_indices = defaultdict(list)
            for idx, doc in enumerate(dataset.documents):
                dataset.content_type_indices[doc['content_type']].append(idx)
        
        logger.info(f"Train set: {len(train_docs)} documents")
        logger.info(f"Val set: {len(val_docs)} documents")
        
        return train_dataset, val_dataset
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        
        # Tokenize with current curriculum length
        tokens = self.tokenizer.encode(doc['text'], max_length=self.current_max_len)
        
        # Ensure minimum length
        if len(tokens) < self.min_seq_len:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.min_seq_len - len(tokens))
        
        # Create input/target pairs
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # Pad to current max length
        pad_length = self.current_max_len - 1
        if len(input_ids) < pad_length:
            padding = [self.tokenizer.pad_token_id] * (pad_length - len(input_ids))
            input_ids = input_ids + padding
            labels = labels + padding
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'content_type': doc['content_type'],
            'quality_score': doc['quality_score'],
            'length': len(tokens)
        }

class ModelTrainer:
    """Enhanced trainer with all improvements"""
    
    def __init__(self, config: ValidationConfig, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = create_validation_model(config)
        self.model.to(self.device)
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def create_optimizer_and_scheduler(self, num_training_steps: int, 
                                     learning_rate: float = 5e-5,
                                     warmup_steps: int = 1000):
        """Create optimizer with learning rate scheduling"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine schedule with warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        logger.info(f"Created optimizer with LR={learning_rate}, warmup={warmup_steps}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, 
                   accumulation_steps: int = 4):
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = defaultdict(default_metrics_dict)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(input_ids, images=batch.get('images', None))
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('prediction_logits'))
                else:
                    logits = outputs
                
                # Compute loss
                loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
                    logits.view(-1, self.config.extended_vocab_size),
                    labels.view(-1)
                )
                
                # Scale loss for accumulation
                loss = loss / accumulation_steps
            
            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            # Track metrics
            batch_loss = loss.item() * accumulation_steps
            epoch_loss += batch_loss
            
            # Track per content type
            for i, content_type in enumerate(batch['content_type']):
                epoch_metrics[content_type]['loss'] += batch_loss
                epoch_metrics[content_type]['count'] += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log periodically
            if batch_idx % 100 == 0:
                self._log_training_progress(batch_idx, epoch, batch_loss, 
                                          current_lr, batch, epoch_metrics)
        
        # Compute epoch averages
        avg_epoch_loss = epoch_loss / len(train_loader)
        for content_type in epoch_metrics:
            count = epoch_metrics[content_type]['count']
            if count > 0:
                epoch_metrics[content_type]['avg_loss'] = (
                    epoch_metrics[content_type]['loss'] / count
                )
        
        return avg_epoch_loss, epoch_metrics
    
    def validate(self, val_loader: DataLoader):
        """Validate model performance"""
        self.model.eval()
        val_loss = 0.0
        val_metrics = defaultdict(default_metrics_dict)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast():
                    outputs = self.model(input_ids)
                    
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('prediction_logits'))
                    else:
                        logits = outputs
                    
                    loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
                        logits.view(-1, self.config.extended_vocab_size),
                        labels.view(-1)
                    )
                
                batch_loss = loss.item()
                val_loss += batch_loss
                
                # Track per content type
                for content_type in batch['content_type']:
                    val_metrics[content_type]['loss'] += batch_loss
                    val_metrics[content_type]['count'] += 1
        
        # Compute averages
        avg_val_loss = val_loss / len(val_loader)
        for content_type in val_metrics:
            count = val_metrics[content_type]['count']
            if count > 0:
                val_metrics[content_type]['avg_loss'] = (
                    val_metrics[content_type]['loss'] / count
                )
        
        return avg_val_loss, val_metrics
    
    def generate_samples(self, prompts: List[str], max_length: int = 50,
                        temperature: float = 0.8, top_p: float = 0.9):
        """Generate text samples for quality assessment"""
        self.model.eval()
        generated_samples = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                input_ids = torch.tensor(
                    [self.tokenizer.encode(prompt, max_length=50)],
                    device=self.device
                )
                
                generated = self._generate_with_sampling(
                    input_ids, max_length, temperature, top_p
                )
                
                # Decode
                generated_text = self.tokenizer.decode(
                    generated[0].tolist(), skip_special=True
                )
                
                generated_samples.append({
                    'prompt': prompt,
                    'generated': generated_text
                })
        
        return generated_samples
    
    def _generate_with_sampling(self, input_ids: torch.Tensor, max_length: int,
                               temperature: float, top_p: float):
        """Generate with nucleus sampling"""
        generated = input_ids
        
        for _ in range(max_length):
            with autocast():
                outputs = self.model(generated)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('prediction_logits'))
                else:
                    logits = outputs
                
                next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return generated
    
    def _log_training_progress(self, batch_idx: int, epoch: int, loss: float,
                              lr: float, batch: Dict, metrics: Dict):
        """Log detailed training progress"""
        # Count content types in batch
        content_counts = Counter(batch['content_type'])
        
        logger.info(f"Epoch {epoch}, Step {batch_idx}: "
                   f"loss={loss:.4f}, lr={lr:.2e}")
        logger.info(f"  Batch composition: {dict(content_counts)}")
        
        # Log per-type metrics
        for content_type, metric in metrics.items():
            if metric['count'] > 0:
                avg_loss = metric['loss'] / metric['count']
                logger.info(f"  {content_type} avg loss: {avg_loss:.4f}")
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float, 
                       metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'config': self.config,
            'is_best': is_best,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }
        
        torch.save(checkpoint, path)
        logger.info(f"💾 Saved checkpoint: {path}")
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 Saved best model: {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        return checkpoint['epoch'], checkpoint['val_loss']

def find_latest_checkpoint(checkpoint_dir: str):
    """Find the latest regular checkpoint (not best checkpoint)"""
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_epoch_') and f.endswith('.pt'):
            # Skip "best" files - only get regular checkpoints
            if '_best.pt' not in f:
                try:
                    # Extract epoch number
                    epoch_str = f.split('_epoch_')[1].split('.pt')[0]
                    epoch_num = int(epoch_str)
                    checkpoint_files.append((f, epoch_num))
                except (ValueError, IndexError):
                    continue
    
    if not checkpoint_files:
        return None, 0
    
    # Find the latest checkpoint
    latest_file, latest_epoch = max(checkpoint_files, key=lambda x: x[1])
    latest_path = os.path.join(checkpoint_dir, latest_file)
    
    return latest_path, latest_epoch

def main():
    """Main training function"""
    # Configuration
    config = ValidationConfig()
    
    # Paths"E:/AI_Training_Data/processed_data"
    general80_dir = "E:/AI/Training Data/General80"
    literature_dir = "E:/AI_Training_Data/processed_data"
    checkpoint_dir = "models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training parameters
    batch_size = 8
    accumulation_steps = 4  # Effective batch size = 32
    num_epochs = 100
    learning_rate = 5e-5
    warmup_ratio = 0.1
    val_ratio = 0.1
    patience = 3  # Early stopping patience
    
    # Create tokenizer
    tokenizer = HaikuSPMTokenizer('haiku_spm.model')

     # Update config with actual tokenizer values
    config.vocab_size = tokenizer.vocab_size
    config.extended_vocab_size = tokenizer.vocab_size
    config.reasoning_tokens = tokenizer.reasoning_tokens
    config.think_start_id = tokenizer.think_start_id
    config.think_end_id = tokenizer.think_end_id

    print(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")

    

    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = EducationalDataset(
        general80_dir=general80_dir,
        literature_dir=literature_dir,
        tokenizer=tokenizer,
        max_seq_len=650,
        min_seq_len=250,
        general80_ratio=0.95,
        quality_threshold=0.65,
        augment_general80=True
    )
    
    # Save tokenizer
    #tokenizer.save(os.path.join(checkpoint_dir, 'tokenizer.pkl'))
    
    # Create train/val split
    train_dataset, val_dataset = dataset.create_train_val_split(val_ratio)
    
    # Create data loaders with balanced sampling
    train_sampler = BalancedBatchSampler(train_dataset, batch_size, general80_ratio=0.95)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 1,  # Larger batch for validation
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Calculate training steps
    steps_per_epoch = len(train_loader) // accumulation_steps
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Create trainer
    trainer = ModelTrainer(config)
    trainer.tokenizer = tokenizer  # Set tokenizer for loss computation
    trainer.create_optimizer_and_scheduler(total_steps, learning_rate, warmup_steps)
    
    # 🔁 AUTO-RESUME FROM LATEST CHECKPOINT
    start_epoch = 0
    latest_checkpoint, latest_epoch = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint:
        logger.info(f"🔁 Found checkpoint: {os.path.basename(latest_checkpoint)}")
        resume = input(f"Resume from epoch {latest_epoch}? (y/n): ").lower().strip()
        
        if resume == 'y':
            start_epoch, _ = trainer.load_checkpoint(latest_checkpoint)
            
            # Handle continued training beyond original schedule
            if start_epoch >= num_epochs:
                logger.info(f"🔄 Detected completed training at epoch {start_epoch}")
                continue_more = input(f"Continue training for more epochs? (y/n): ").lower().strip()
                
                if continue_more == 'y':
                    # Reset learning rate for continued training
                    new_lr = 1e-4  # Lower than original 5e-5
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    # Create new scheduler for remaining epochs
                    remaining_epochs = num_epochs - start_epoch
                    remaining_steps = steps_per_epoch * remaining_epochs
                    trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        trainer.optimizer, 
                        T_max=remaining_steps,
                        eta_min=1e-6
                    )
                    
                    logger.info(f"✅ Reset LR to {new_lr} for {remaining_epochs} more epochs")
                else:
                    logger.info("🏁 Keeping existing model as final")
                    exit()
            
            logger.info(f"✅ Resumed from epoch {start_epoch}, best_val_loss={trainer.best_val_loss:.4f}")
        else:
            logger.info("🆕 Starting fresh training")
    else:
        logger.info("🆕 No checkpoint found, starting fresh training")

    # Test prompts for generation
    test_prompts = [
        # Explanatory style
        " Human: What is 7 + 5?  <answer>  </answer>  ",
        " the sun is...? ",
        " Human: What is 1 + 1?  <answer>   </answer> ",
        " Human: How does a car windshield wiper work?  <answer>    ?   </answer>",
        
        # Narrative style
        "Once upon a time in a small village",
        "The student walked into the classroom",
        
        # Instructional style
        "First, we need to identify",
        "The process begins when",
        "To solve this problem, consider"
    ]
    
    logger.info("🚀 Starting training...")
    logger.info(f"Total epochs: {num_epochs}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Effective batch size: {batch_size * accumulation_steps}")
    
    # Training loop with auto-resume
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        # Update curriculum learning
        train_dataset.set_curriculum_length(epoch, num_epochs)
        val_dataset.current_max_len = train_dataset.current_max_len
        
        # Train
        train_loss, train_metrics = trainer.train_epoch(train_loader, epoch, accumulation_steps)
        
        # Validate
        val_loss, val_metrics = trainer.validate(val_loader)
        
        # Log epoch results
        logger.info(f"\n📊 Epoch {epoch + 1} Results:")
        logger.info(f"  Train loss: {train_loss:.4f}")
        logger.info(f"  Val loss: {val_loss:.4f}")
        
        for content_type in ['explanatory', 'literature']:
            if content_type in train_metrics:
                train_avg = train_metrics[content_type].get('avg_loss', 0)
                val_avg = val_metrics[content_type].get('avg_loss', 0)
                logger.info(f"  {content_type}: train={train_avg:.4f}, val={val_avg:.4f}")
        
        # Generate samples
        if epoch % 2 == 0:  # Every 2 epochs
            logger.info("\n📝 Sample generations:")
            samples = trainer.generate_samples(test_prompts[:3], max_length=30)
            for sample in samples:
                logger.info(f"  Prompt: {sample['prompt']}")
                logger.info(f"  Generated: {sample['generated'][:100]}...")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt'
        )
        is_best = val_loss < trainer.best_val_loss
        
        trainer.save_checkpoint(
            checkpoint_path, epoch + 1, val_loss, 
            {'train': train_metrics, 'val': val_metrics},
            is_best
        )
        
        # Early stopping
        if is_best:
            trainer.best_val_loss = val_loss
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1
            
        if trainer.patience_counter >= patience:
            logger.info(f"⚠️ Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Final evaluation
    logger.info("\n🏁 Training completed!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Generate final samples
    logger.info("\n📝 Final sample generations:")
    final_samples = trainer.generate_samples(test_prompts, max_length=50)
    for sample in final_samples:
        logger.info(f"\nPrompt: {sample['prompt']}")
        logger.info(f"Generated: {sample['generated']}")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'final_model.pt')
    trainer.save_checkpoint(
        final_path, num_epochs, trainer.best_val_loss,
        {'final': True}, is_best=False
    )

if __name__ == "__main__":
    main()