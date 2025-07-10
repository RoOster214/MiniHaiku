# spm_tokenizer.py
import sentencepiece as spm
from typing import List, Optional, Dict
import torch
import os

class HaikuSPMTokenizer:
    """SentencePiece tokenizer wrapper for Haiku model"""
    
    def __init__(self, model_path: str = 'haiku_spm.model'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
            
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp.vocab_size()
        
        # Special token IDs
        self.pad_token_id = 0  # <pad>
        self.unk_token_id = 1  # <unk>
        self.bos_token_id = 2  # <bos>
        self.eos_token_id = 3  # <eos>
        
        # Load reasoning tokens
        self.reasoning_tokens = {}
        reasoning_token_names = [
            '<think>', '</think>',
            '<step1>', '<step2>', '<step3>', '<step4>', 
            '<step5>', '<step6>', '<step7>', '<step8>',
            '<answer>', '</answer>',
            '<branch1>', '<branch2>', '<branch3>',
            '<approach>', '<assessment>', '<synthesis>',
            '<verification>', '<reflection>', '<correction>', '<retry>',
            '<analysis>', '<hypothesis>', '<evidence>', '<conclusion>'
        ]
        
        for token in reasoning_token_names:
            token_id = self.sp.piece_to_id(token)
            if token_id != self.unk_token_id:  # Token exists
                self.reasoning_tokens[token] = token_id
        
        # Set specific token IDs for compatibility
        self.think_start_id = self.reasoning_tokens.get('<think>')
        self.think_end_id = self.reasoning_tokens.get('</think>')
        self.step_start_id = self.reasoning_tokens.get('<step1>')
        self.step_end_id = self.reasoning_tokens.get('<step8>')
        
        print(f"Loaded tokenizer with {self.vocab_size} tokens")
        print(f"Found {len(self.reasoning_tokens)} reasoning tokens")
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.sp.encode(text, out_type=int)
        
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
            
        return tokens
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        
        if skip_special:
            token_ids = [t for t in token_ids if t != self.pad_token_id]
        
        try:
            return self.sp.decode(token_ids)
        except:
            return ""
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def __len__(self):
        return self.vocab_size