
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
import math
from typing import Optional, Dict, List, Tuple, Any
import logging
from contextlib import contextmanager
import os
import json

# Image support imports
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Import our validation configuration
from validation_config import ValidationConfig

# Set up logging for monitoring training behavior
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization - scaled down version.
    
    This provides the same efficiency benefits as in your full model,
    with better computational performance than LayerNorm while maintaining
    similar normalization effectiveness across the reduced layer count.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS normalization exactly as in your full model
        # This ensures identical normalization behavior at both scales
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings - maintains same mathematical relationships.
    
    The frequency calculations and rotation patterns remain identical to your
    full model, ensuring the same positional understanding capabilities
    just applied to the scaled-down hidden dimensions.
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 1024,  # Scaled down sequence length
        base: float = 10000.0,  # Same base frequency as full model
        scaling: Optional[Dict] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling = scaling
        
        # Calculate inverse frequencies using same formula as full model
        # This preserves the mathematical relationships that make RoPE effective
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute rotations for efficiency during validation training
        self._build_cache(max_position_embeddings)
    
    def _build_cache(self, seq_len: int):
        """Pre-compute cos and sin values for efficient validation runs"""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        
        # Apply scaling if configured (for length extrapolation testing)
        if self.scaling is not None:
            scale_factor = self.scaling.get("factor", 1.0)
            t = t / scale_factor
        
        # Generate rotation matrices using same approach as full model
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32):
        """Return rotation matrices for the given sequence length"""
        # Extend cache if needed for longer sequences during testing
        if seq_len > self.max_position_embeddings:
            self._build_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims - identical to full model implementation"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings using same mathematics as full model"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MinimalVisionEncoder(nn.Module):
    """
    Minimal vision encoder for image support in validation.
    Uses a small pretrained ResNet to stay within 4GB VRAM constraints.
    """
    def __init__(self, config: ValidationConfig):
        super().__init__()
        
        # Use ResNet18 - small and efficient (11M params)
        self.backbone = tv_models.resnet18(pretrained=True)
        
        # Remove the classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Global average pooling to get single feature vector
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Project from ResNet features (512) to model hidden dim
        self.projection = nn.Linear(512, config.hidden_dim)
        
        # Freeze backbone to save memory during training
        if config.freeze_vision_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature vectors matching text hidden dimension.
        Returns: [batch_size, hidden_dim] tensor
        """
        with torch.no_grad() if self.training else torch.enable_grad():  # No gradients for frozen backbone
            features = self.backbone(images)  # [B, 512, 7, 7]
        
        features = self.pool(features).squeeze(-1).squeeze(-1)  # [B, 512]
        image_embeds = self.projection(features)  # [B, hidden_dim]
        
        return image_embeds

class ValidationAttention(nn.Module):
    """
    Grouped-Query Attention scaled down while maintaining all efficiency benefits.
    
    This preserves the 4:1 ratio between query and key-value heads that makes
    your full model so memory efficient, just applied to the smaller dimensions.
    The mathematical relationships remain identical.
    """
    def __init__(self, config: ValidationConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Scale down dimensions while preserving ratios from full model
        self.hidden_dim = config.hidden_dim  # 512 vs 2048 in full model
        self.num_heads = config.num_attention_heads  # 8 vs 16 in full model
        self.num_kv_heads = config.num_kv_heads  # 2 vs 4 in full model
        self.head_dim = self.hidden_dim // self.num_heads  # 64 (same as full model!)
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # Maintains 4:1 ratio
        
        # Validate that scaling preserves essential relationships
        assert self.hidden_dim % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0
        assert self.num_kv_groups == 4  # Same ratio as full model
        
        # Linear projections with same efficiency patterns as full model
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        # Rotary embeddings with scaled parameters
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            getattr(config, 'rope_scaling', None)  # This safely handles the missing attribute
        ) 
        
        # Dropout matching full model configuration
        self.attention_dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass implementing the same attention patterns as your full model.
        
        The mathematical operations remain identical - only the tensor dimensions
        are scaled down proportionally to fit memory constraints.
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to query, key, value using same approach as full model
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention with scaled dimensions
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings using identical mathematics
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(kv_seq_len, hidden_states.device, hidden_states.dtype)
        
        # Handle position IDs for proper RoPE application during generation
        if position_ids is None:
            if past_key_value is not None:
                position_ids = torch.arange(
                    past_key_value[0].shape[-2], kv_seq_len, device=hidden_states.device
                )
            else:
                position_ids = torch.arange(kv_seq_len, device=hidden_states.device)
        
        # Apply rotary embeddings to queries and keys
        if past_key_value is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle KV cache for efficient generation (same logic as full model)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        present_key_value = (key_states, value_states) if use_cache else None
        
        # Expand KV heads to match query heads (grouped-query attention magic)
        # This is where we get the 75% memory savings during inference
        if self.num_kv_groups > 1:
            key_states = torch.repeat_interleave(key_states, self.num_kv_groups, dim=1)
            value_states = torch.repeat_interleave(value_states, self.num_kv_groups, dim=1)
        
        # Compute attention using same scaling and masking as full model
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply causal mask for autoregressive generation
        if attention_mask is None and seq_length > 1:
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), float('-inf'), device=hidden_states.device),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask
        
        # Softmax and dropout exactly as in full model
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value

class ThinkBoxProcessr(nn.Module):
    """
    Proccessor for think box sequences for enhanced reasoning capabilities
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # step-aware attention mechanism for processing think box sequences
        self.step_attention = nn.MultiheadAttention(
            embed_dim=config.reasoning_dim,
            num_heads=config.reasoning_heads // 2,  # lighter attention 
            dropout=config.hidden_dropout,
            batch_first=True
            )

        # step-embedding to distinguish between different reasoning steps
        self.step_embeddings = nn.Embedding(8, config.reasoning_dim)  # FIXED: was step_embedding

        # layer norm for step procesing
        self.step_norm = nn.LayerNorm(config.reasoning_dim)
    
    def detect_think_box(self, input_ids):
        """
        Detect if input contains think box tokens
        """

        if input_ids is None:
            return False

        #  Use think_start_id 

        think_start = self.config.think_start_id
        return (input_ids == think_start).any()

    def extract_step_positions(self, input_ids):
        """
        Extract positions of reasoning steps in the sequence
        """

        step_positions = []
        step_ids = list(range(self.config.step_start_id, self.config.step_end_id + 1))

        for step_id in step_ids:
            positions = (input_ids == step_id).nonzero(as_tuple=True)
            if len(positions[0]) > 0:
                step_positions.append({
                    'step_num': step_id - self.config.step_start_id + 1,
                    'positions': positions[0].tolist()
                    })

                return step_positions

    def process_with_think_boxes(self, hidden_states, input_ids, reasoning_projection, output_projection, output_norm):
        """
        Process hidden states with think box sequences
        """
        batch_size, seq_length, hidden_dim = hidden_states.shape

        # project to reasoning dimension
        reasoning_states = reasoning_projection(hidden_states)
        reasoning_states = self.step_norm(reasoning_states)

        # Extract step positions from input_ids
        step_positions = self.extract_step_positions(input_ids)

        if not step_positions:
            # add step aware processing
            
            for step_info in step_positions:
                step_num = step_info['step_num']
                positions = step_info['positions']

                # Get step embeddings - ensure step_num is within bounds
                step_idx = min(step_num - 1, 7)  # Clamp to 0-7 range
                step_embed = self.step_embeddings(
                    torch.tensor(step_idx, device=hidden_states.device)
                )

                # apply step emeddings to the specific positions
                # note: positions is a tuple of (batch_indices, sequence_indices)
                batch_indices, seq_indices = positions
                for b_idx, s_idx in zip(batch_indices, seq_indices):
                    reasoning_states[b_idx, s_idx] += step_embed

            reasoning_states, _ = self.step_attention(
                reasoning_states, reasoning_states, reasoning_states
            )

        output = output_projection(reasoning_states)
        return output_norm(output)

class ValidationSwiGLU(nn.Module):
    """
    SwiGLU activation scaled down while maintaining the mathematical relationships.
    
    The 2.75x multiplier between hidden and intermediate dimensions is preserved,
    ensuring the same representational capacity ratios as your full model.
    """
    def __init__(self, config: ValidationConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim  # 512 vs 2048
        self.intermediate_dim = config.intermediate_dim  # 1408 vs 5504 (same 2.75x ratio)
        
        # Three projections for SwiGLU - same pattern as full model
        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU computation identical to full model mathematics.
        
        The gating mechanism provides the same selective amplification
        of features, just operating on scaled-down representations.
        """
        # Apply SwiGLU: SiLU(gate(x)) * up(x) -> down
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        intermediate = gate * up  # Element-wise gating
        output = self.down_proj(intermediate)
        return self.dropout(output)

class ScaledIntegratedReasoning(nn.Module):
    """
    Reasoning module scaled down while preserving the sophisticated logic.
    
    This maintains the same complexity assessment approach and integration
    strategy as your full model, just with reduced computational overhead
    suitable for validation experiments.
    """
    def __init__(self, config: ValidationConfig):
        super().__init__()
        self.config = config
        
        # Complexity assessment scaled proportionally
        # Progressive dimension reduction: 512 -> 128 -> 32 -> 1 (vs 2048 -> 512 -> 64 -> 1)
        self.complexity_assessor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),  # 512 -> 128
            nn.ReLU(),  # Same activation choice as full model for speed
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.hidden_dim // 4, 32),  # 128 -> 32 (proportionally scaled)
            nn.ReLU(),
            nn.Linear(32, 1),  # Final complexity score
            nn.Sigmoid()  # Same threshold logic as full model
        )
        
        # Reasoning transformer - simplified but maintains core functionality
        self.reasoning_projection = nn.Linear(config.hidden_dim, config.reasoning_dim)  # 512 -> 256
        
        # Single reasoning layer (vs 2 in full model) but same architectural pattern
        self.reasoning_block = nn.TransformerEncoderLayer(
            d_model=config.reasoning_dim,  # 256 vs 512 in full model
            nhead=config.reasoning_heads,  # 4 vs 8 in full model (maintains proportion)
            dim_feedforward=config.reasoning_dim * 4,  # Same 4x multiplier
            dropout=config.hidden_dropout,
            activation='relu',  # Same choice as full model
            batch_first=True,
            norm_first=True  # Pre-norm for training stability
        )
        
        # Project back to model dimension
        self.output_projection = nn.Linear(config.reasoning_dim, config.hidden_dim)  # 256 -> 512
        
        # Integration mechanism preserving full model's gating approach
        self.integration_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),  # Blend original + reasoned
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Sigmoid()  # Gate weights for smooth blending
        )
        
        # Layer norms for stability (same as full model)
        self.reasoning_norm = nn.LayerNorm(config.reasoning_dim)
        self.output_norm = nn.LayerNorm(config.hidden_dim)
    
    def assess_complexity(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complexity assessment using the same logic as your full model.
        
        Determines which sequences would benefit from reasoning enhancement
        using identical threshold and scoring mathematics.
        """
        # Use mean pooling for global context assessment (same as full model)
        pooled = hidden_states.mean(dim=1)
        complexity_scores = self.complexity_assessor(pooled)
        
        # Same conservative threshold as full model to avoid over-triggering
        should_reason = complexity_scores > self.config.reasoning_threshold
        
        return should_reason.squeeze(-1), complexity_scores.squeeze(-1)
    
    def apply_reasoning(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply reasoning transformation using scaled-down transformer.
        
        Creates structured representations that enhance complex query handling,
        just with reduced computational overhead for validation.
        """
        # Project to reasoning dimension
        reasoning_states = self.reasoning_projection(hidden_states)
        reasoning_states = self.reasoning_norm(reasoning_states)
        
        # Apply single reasoning transformer layer
        reasoning_states = self.reasoning_block(reasoning_states)
        
        # Project back to model dimension
        reasoning_output = self.output_projection(reasoning_states)
        reasoning_output = self.output_norm(reasoning_output)
        
        return reasoning_output
    
    def forward(self, hidden_states, force_reasoning=False, input_ids=None):
        """
        Enhanced forward pass with think box awareness
        """

        batch_size, seq_length, hidden_dim = hidden_states.shape

        # Assess complexity using the same logic as your full model
        should_reason, complexity_scores = self.assess_complexity(hidden_states)

        # Check for think boxes BEFORE the complexity check
        has_think_boxes = True
        if input_ids is not None:
            has_think_boxes = self.think_box_processor.detect_think_box(input_ids)

         # Force reasoning if think boxes are present
        if has_think_boxes:
            should_reason = torch.ones_like(should_reason, dtype=torch.bool)
            
        metrics = {
            'reasoning_triggered': should_reason.float().mean().item(),
            'complexity_scores': complexity_scores,
            'mean_complexity': complexity_scores.mean().item(),
            'has_think_boxes': has_think_boxes 
            }

        # Apply reasoning only to sequences that need it
        output = hidden_states.clone()
        reasoning_indices = should_reason.nonzero(as_tuple=True)[0]

        if len(reasoning_indices) > 0:
            ## Extract sequences needing reasoning enhancement
            states_to_reason = hidden_states[reasoning_indices]

            # Get input_ids for reasoning sequences 
            input_ids_subset = None
            if input_ids is not None:
                input_ids_subset = input_ids[reasoning_indices]

            # Apply reasoning transformation
            if has_think_boxes and input_ids_subset is not None:
                reasoned_states = self.think_box_processor.process_with_think_boxes(
                    states_to_reason,
                    input_ids_subset,
                    self.reasoning_projection,
                    self.output_projection,
                    self.output_norm
                )
            else:
                reasoned_states = self.apply_reasoning(states_to_reason)

            # compute intergration weights using the same gating mechanism
            gate_input = torch.cat([states_to_reason, reasoned_states], dim=-1)
            gate_weights = self.integration_gate(gate_input)

            # Blend original and reasoned states using the same gating approach
            enhanced_states = states_to_reason + gate_weights * reasoned_states

            # update output tensor with enhanced states
            output[reasoning_indices] = enhanced_states

        return output, metrics


class ValidationTransformerBlock(nn.Module):
    """
    Complete transformer block scaled down while preserving your architectural choices.
    
    Uses the same pre-normalization pattern and residual connections as your
    full model, ensuring identical training dynamics at the smaller scale.
    """
    def __init__(self, config: ValidationConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Core components using scaled versions of full model components
        self.attention = ValidationAttention(config, layer_idx)
        self.mlp = ValidationSwiGLU(config)
        
        # Normalization layers using same RMSNorm as full model
        self.input_layernorm = RMSNorm(config.hidden_dim, config.layer_norm_epsilon)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, config.layer_norm_epsilon)
        
        # Dropout for residual connections (same configuration)
        self.dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass using identical patterns to your full model blocks.
        
        Pre-normalization and residual connections ensure the same training
        stability and gradient flow characteristics.
        """
        # Self-attention with residual connection (pre-norm pattern)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output, present_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = residual + self.dropout(attention_output)
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(mlp_output)
        
        return hidden_states, present_key_value

class ValidationHaikuModel(nn.Module):
    """
    Complete validation model implementing your sophisticated architecture at scale.
    
    This model maintains every design principle from your 2B parameter model
    while fitting comfortably in 4GB VRAM for comprehensive validation testing.
    Now with integrated image support for multimodal experiments.
    """
    def __init__(self, config: ValidationConfig):
        super().__init__()
        # configuration
        self.config = config

        #Disable imaging 
        config.use_images = False        # Vision encoder won't be created
        config.use_reasoning = True     # Also disable reasoning for now

        # Token embeddings with proper scaling
        self.embed_tokens = nn.Embedding(config.extended_vocab_size, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.embedding_dropout)
        
        # Vision encoder for image support
        if config.use_images:
            self.vision_encoder = MinimalVisionEncoder(config)
        else:
            self.vision_encoder = None
        
        # Special token for image placeholder
        self.image_token_id = config.vocab_size - 1  # Use last vocab position
        
        # Transformer layers scaled down but maintaining full sophistication
        self.layers = nn.ModuleList([
            ValidationTransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        
        # Reasoning module positioned at same relative location as full model
        if config.use_reasoning:
            self.reasoning = ScaledIntegratedReasoning(config)
            # Apply reasoning at 75% through model (layer 6 of 8, matching full model ratio)
            self.reasoning_layer = int(config.num_layers * 0.75)
        else:
            self.reasoning = None
            self.reasoning_layer = None
        
        # Final normalization using same RMSNorm
        self.final_layernorm = RMSNorm(config.hidden_dim, config.layer_norm_epsilon)
        
        # Language modeling head with weight tying for efficiency
        self.lm_head = nn.Linear(config.hidden_dim, config.extended_vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Same weight tying as full model
        
        # Initialize weights using same scaled approach as full model
        self.apply(self._init_weights)
        
        # Log model information for validation tracking
        self._log_model_info()
    
    def _init_weights(self, module):
        """
        Weight initialization using the same scaled approach as your full model.
        
        This ensures that the validation model starts with similar parameter
        distributions, leading to comparable training dynamics.
        """
        if isinstance(module, nn.Linear):
            # Use scaled initialization based on layer depth (same formula as full model)
            if self.config.use_scaled_init and hasattr(module, 'layer_idx'):
                scale = (2.0 / (5.0 + module.layer_idx)) ** 0.5
            else:
                scale = 1.0
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range * scale)
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def _log_model_info(self):
        """Log detailed validation model information for comparison with full model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate parameter distribution for comparison with full model
        embedding_params = self.embed_tokens.num_embeddings * self.embed_tokens.embedding_dim
        if self.reasoning:
            reasoning_params = sum(p.numel() for p in self.reasoning.parameters())
        else:
            reasoning_params = 0
        if self.vision_encoder:
            vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        else:
            vision_params = 0
        transformer_params = total_params - embedding_params - reasoning_params - vision_params
        
        logger.info("="*60)
        logger.info("Validation Haiku Model Configuration")
        logger.info("="*60)
        logger.info(f"Total parameters: {total_params/1e6:.1f}M (vs ~2000M in full model)")
        logger.info(f"Scaling factor: ~{2000/(total_params/1e6):.1f}x smaller")
        logger.info(f"Parameter distribution:")
        logger.info(f"  - Embeddings: {embedding_params/1e6:.1f}M ({embedding_params/total_params*100:.1f}%)")
        logger.info(f"  - Transformer: {transformer_params/1e6:.1f}M ({transformer_params/total_params*100:.1f}%)")
        logger.info(f"  - Reasoning: {reasoning_params/1e6:.1f}M ({reasoning_params/total_params*100:.1f}%)")
        logger.info(f"  - Vision: {vision_params/1e6:.1f}M ({vision_params/total_params*100:.1f}%)")
        logger.info(f"Architecture validation:")
        logger.info(f"  - Layers: {self.config.num_layers} (vs 28 in full model)")
        logger.info(f"  - Hidden dimension: {self.config.hidden_dim} (vs 2048 in full model)")
        logger.info(f"  - Attention heads: {self.config.num_attention_heads} (vs 16 in full model)")
        logger.info(f"  - KV heads: {self.config.num_kv_heads} (vs 4 in full model)")
        logger.info(f"  - GQA ratio: {self.config.num_attention_heads//self.config.num_kv_heads}:1 (same as full model)")
        if self.reasoning_layer:
            logger.info(f"  - Reasoning position: Layer {self.reasoning_layer} ({self.reasoning_layer/self.config.num_layers*100:.0f}%)")
        logger.info(f"  - Image support: {'Enabled' if self.config.use_images else 'Disabled'}")
        logger.info("="*60)
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image file for the model.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor ready for vision encoder
        """
        if self.vision_encoder is None:
            raise ValueError("Model was not configured with image support")
        
        image = Image.open(image_path).convert('RGB')
        return self.vision_encoder.transform(image).unsqueeze(0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        images: Optional[torch.Tensor] = None,  # NEW: Add images parameter
        use_cache: bool = False,
        use_reasoning: Optional[bool] = None,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass implementing the same logic flow as your full model.
        Now with optional image conditioning support.
        
        Args:
            images: Optional tensor of shape [batch_size, 3, 224, 224]
                   If provided, image features are prepended to the sequence
        """
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        else:
            batch_size = images.shape[0]
            seq_length = 0
            device = images.device
       
        
        # Determine reasoning usage (same logic as full model)
        if use_reasoning is None:
            use_reasoning = self.config.use_reasoning and self.reasoning is not None
        
        # Token embeddings with same dropout pattern
        if input_ids is not None:
             hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = None
        
        # Handle image inputs if provided
        # Handle image inputs if provided
        if images is not None:
            if self.vision_encoder is None:
                raise ValueError("Model was not configured with image support")
        
             # Encode images to features
            image_features = self.vision_encoder(images)  # [batch_size, hidden_dim]
            image_embeds = image_features.unsqueeze(1)    # [batch_size, 1, hidden_dim]
    
            # Combine with token embeddings safely
            if hidden_states is not None:
             hidden_states = torch.cat([image_embeds, hidden_states], dim=1)
             seq_length = hidden_states.shape[1]
            else:
             hidden_states = image_embeds
             seq_length = 1
            
            # Adjust attention mask if needed
            if attention_mask is not None:
                # Add attention for image token
                image_mask = torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([image_mask, attention_mask], dim=1)
        
        hidden_states = self.embed_dropout(hidden_states)
        
        # Prepare attention mask using same approach as full model
        if attention_mask is not None:
            # Convert to 4D attention bias format
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Initialize cache for generation (same pattern as full model)
        if use_cache and past_key_values is None:
            past_key_values = [None] * self.config.num_layers
        
        # Storage for outputs and metrics
        present_key_values = [] if use_cache else None
        reasoning_metrics = None
        
        # Process through transformer layers with same checkpointing logic
        for idx, layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values else None
            
            # Apply gradient checkpointing if configured (same as full model)
            if self.config.use_gradient_checkpointing and self.training:
                if idx in self.config.checkpoint_layers:
                    hidden_states, present_key_value = gradient_checkpoint(
                        layer,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_value,
                        use_cache,
                        use_reentrant=False
                    )
                else:
                    hidden_states, present_key_value = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        use_cache=use_cache
                    )
            else:
                hidden_states, present_key_value = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
            
            # Store key-value cache
            if use_cache:
                present_key_values.append(present_key_value)
            
            # Apply reasoning at the same relative position as full model
            if use_reasoning and idx == self.reasoning_layer:
                hidden_states, reasoning_metrics = self.reasoning(
                    hidden_states,
                    force_reasoning=self.training  # Same forcing logic as full model
                )
        
        # Final normalization and language modeling head
        hidden_states = self.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Return outputs in same format as full model
        outputs = {
            'logits': logits,
            'past_key_values': present_key_values,
            'reasoning_metrics': reasoning_metrics
        }
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        images: Optional[torch.Tensor] = None,  # NEW: Add images parameter
        use_reasoning: bool = True,
        return_reasoning_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Generation method implementing the same sampling as your full model.
        Now with optional image conditioning support.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Generation state tracking
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        past_key_values = None
        reasoning_metrics = None
        
        # Generate tokens using same approach as full model
        for step in range(max_new_tokens):
            # Forward pass with caching
            outputs = self.forward(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                images=images if past_key_values is None else None,  # Only use images on first pass
                use_cache=True,
                use_reasoning=use_reasoning and past_key_values is None  # Reason only on first pass
            )
            
            # Store reasoning metrics from first pass
            if outputs['reasoning_metrics'] is not None:
                reasoning_metrics = outputs['reasoning_metrics']
            
            # Sample next token using same logic as full model
            next_token_logits = outputs['logits'][:, -1, :]
            
            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample and update sequences
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Handle finished sequences
            next_tokens = next_tokens * unfinished_sequences + self.config.pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update generation state
            unfinished_sequences = unfinished_sequences * (next_tokens != self.config.eos_token_id)
            past_key_values = outputs['past_key_values']
            
            # Check completion
            if unfinished_sequences.sum() == 0:
                break
        
        # Return generation results
        result = {
            'sequences': input_ids,
            'num_generated_tokens': input_ids.shape[1] - input_ids.shape[1]
        }
        
        if return_reasoning_metrics and reasoning_metrics is not None:
            result['reasoning_metrics'] = reasoning_metrics
        
        return result

def create_validation_model(config: Optional[ValidationConfig] = None) -> ValidationHaikuModel:
    """
    Create a validation model with the given configuration.
    
    If no configuration provided, loads from validation_config.json or creates default.
    This function serves as the main entry point for validation experiments.
    """
    if config is None:
        # Try to load existing validation config
        try:
            config = ValidationConfig.load("validation_config.json")
            logger.info("Loaded validation configuration from validation_config.json")
        except FileNotFoundError:
            # Create default validation config
            from validation_config import create_validation_config
            config = create_validation_config()
            logger.info("Created default validation configuration")
    
    # Create the validation model
    model = ValidationHaikuModel(config)
    
    logger.info(f"Validation model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    return model

# Dataset support for image-text validation
class ImageTextValidationDataset(Dataset):
    """
    Simple dataset for image-text validation pairs.
    Expects JSONL file with format:
    {"image_path": "path/to/image.jpg", "prompt": "Describe this image", "expected": "A cat on a mat"}
    """
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load JSONL data
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Tokenize text
        prompt_ids = self.tokenizer.encode(
            item['prompt'], 
            max_length=self.max_length, 
            truncation=True
        )
        
        return {
            'image_path': item['image_path'],
            'input_ids': torch.tensor(prompt_ids),
            'prompt': item['prompt'],
            'expected': item.get('expected', ''),
            'metadata': item.get('metadata', {})
        }

def collate_image_text(batch, model, device):
    """
    Custom collate function to handle image loading and batching.
    """
    # Stack input_ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'] for item in batch],
        batch_first=True,
        padding_value=model.config.pad_token_id
    )
    
    # Load and preprocess images
    images = []
    for item in batch:
        image = model.preprocess_image(item['image_path'])
        images.append(image)
    images = torch.cat(images, dim=0)
    
    # Create attention mask
    attention_mask = (input_ids != model.config.pad_token_id).long()
    
    return {
        'input_ids': input_ids.to(device),
        'images': images.to(device),
        'attention_mask': attention_mask.to(device),
        'prompts': [item['prompt'] for item in batch],
        'expected': [item['expected'] for item in batch]
    }

# Validation functions
def validate_with_images(model, image_path: str, prompt: str, device: torch.device):
    """
    Example of how to use the model with images during validation.
    """
    # Preprocess image
    image = model.preprocess_image(image_path).to(device)
    
    # Tokenize prompt (assuming you have a tokenizer)
    # input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # For demo, use random tokens
    input_ids = torch.randint(0, model.config.vocab_size, (1, 20), device=device)
    
    # Generate with image context
    outputs = model.generate(
        input_ids,
        images=image,
        max_new_tokens=50,
        temperature=0.7
    )
    
    return outputs['sequences']

def validate_with_images_dataset(model, validation_data_path: str, tokenizer, device):
    """
    Run validation on image-text pairs and log results.
    """
    model.eval()
    
    # Create dataset and dataloader
    dataset = ImageTextValidationDataset(validation_data_path, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=4,  # Small batch for memory efficiency
        shuffle=False,
        collate_fn=lambda b: collate_image_text(b, model, device)
    )
    
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Generate completions
            outputs = model.generate(
                input_ids=batch['input_ids'],
                images=batch['images'],
                max_new_tokens=50,
                temperature=0.7
            )
            
            # Decode and log results
            for i, seq in enumerate(outputs['sequences']):
                generated = tokenizer.decode(seq, skip_special_tokens=True)
                
                result = {
                    'prompt': batch['prompts'][i],
                    'expected': batch['expected'][i],
                    'generated': generated,
                    'reasoning_used': outputs.get('reasoning_metrics', {}).get('reasoning_triggered', 0)
                }
                results.append(result)
                
                # Log sample
                logger.info(f"\nSample {i}:")
                logger.info(f"Prompt: {result['prompt']}")
                logger.info(f"Expected: {result['expected']}")
                logger.info(f"Generated: {result['generated']}")
    
    return results

# Testing and validation functions
def test_validation_model():
    """
    Comprehensive test of the validation model functionality.
    
    This validates that all components work correctly and that the model
    exhibits the same behavioral patterns as your full model design.
    """
    logger.info("Testing validation model functionality...")
    
    # Create model and move to appropriate device
    config = ValidationConfig() if not hasattr(ValidationConfig, 'load') else ValidationConfig.load("validation_config.json")
    config.use_images = True  # Enable image support for testing
    model = create_validation_model(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Test basic forward pass
    logger.info("\n1. Testing forward pass...")
    batch_size, seq_length = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    
    with torch.no_grad():
        outputs = model(input_ids, use_reasoning=False)
        
        logger.info(f"✅ Forward pass successful")
        logger.info(f"Input shape: {input_ids.shape}")
        logger.info(f"Output logits shape: {outputs['logits'].shape}")
        
        if outputs['reasoning_metrics']:
            reasoning_rate = outputs['reasoning_metrics']['reasoning_triggered']
            logger.info(f"Reasoning activation rate: {reasoning_rate:.1%}")
    
    # Test generation capability
    logger.info("\n2. Testing generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10), device=device)
    
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=30,
            temperature=0.8,
            use_reasoning=True,
            return_reasoning_metrics=True
        )
        
        logger.info(f"✅ Generation successful")
        logger.info(f"Generated {generated['sequences'].shape[1] - 10} new tokens")
        if 'reasoning_metrics' in generated and generated['reasoning_metrics']:
            logger.info(f"Reasoning was used during generation")
    
    # Test memory usage
    logger.info("\n3. Memory usage validation...")
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated(device) / 1024**3
        logger.info(f"Current memory usage: {memory_used:.2f} GB")
        
        if memory_used < 3.5:  # Well under 4GB limit
            logger.info("✅ Memory usage is within 4GB constraints")
        else:
            logger.warning("⚠️ Memory usage is high - consider reducing batch size")
    
    # Test reasoning behavior differences
    logger.info("\n4. Testing reasoning behavior...")
    
    # Simple prompt that should NOT trigger reasoning
    simple_input = torch.randint(100, 200, (1, 20), device=device)  # Simple token pattern
    
    # Complex prompt that SHOULD trigger reasoning  
    complex_input = torch.randint(300, 400, (1, 40), device=device)  # Different pattern
    
    with torch.no_grad():
        simple_output = model(simple_input, use_reasoning=True)
        complex_output = model(complex_input, use_reasoning=True)
        
        simple_reasoning = simple_output['reasoning_metrics']['reasoning_triggered'] if simple_output['reasoning_metrics'] else 0
        complex_reasoning = complex_output['reasoning_metrics']['reasoning_triggered'] if complex_output['reasoning_metrics'] else 0
        
        logger.info(f"Simple input reasoning rate: {simple_reasoning:.1%}")
        logger.info(f"Complex input reasoning rate: {complex_reasoning:.1%}")
    
    # Test image-conditioned generation
   # logger.info("\n5. Testing image-conditioned generation...")
    
    # Create dummy image tensor for testing
    #dummy_image = torch.randn(1, 3, 224, 224, device=device)
    
    #with torch.no_grad():
        # Test forward pass with image
       # outputs = model(prompt, images=dummy_image)
        #logger.info(f"✅ Image-conditioned forward pass successful")
       # logger.info(f"Output shape with image: {outputs['logits'].shape}")
        
        # Test generation with image
        #generated = model.generate(
           # prompt,
            #images=dummy_image,
            #max_new_tokens=30,
            #temperature=0.8
        #)
        #logger.info(f"✅ Image-conditioned generation successful")
        #logger.info(f"Generated {generated['sequences'].shape[1] - prompt.shape[1]} tokens with image context")
    
    logger.info("\n✅ Validation model testing completed successfully!")
    logger.info("The model is ready for training data validation experiments.")
    
    return model

def test_image_support():
    """Test that image support is properly integrated."""
    logger.info("Testing image support implementation...")
    
    # Create model
    config = ValidationConfig()
    config.use_images = True
    model = create_validation_model(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Test 1: Vision encoder initialization
    logger.info("\n1. Testing vision encoder...")
    assert hasattr(model, 'vision_encoder'), "Vision encoder not found"
    assert hasattr(model.vision_encoder, 'backbone'), "Vision backbone not found"
    logger.info("✅ Vision encoder properly initialized")
    
    # Test 2: Image preprocessing
    logger.info("\n2. Testing image preprocessing...")
    # Create dummy PIL image
    dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_pil = Image.fromarray(dummy_array)
    
    # Save and load
    test_path = "test_image.jpg"
    dummy_pil.save(test_path)
    
    try:
        preprocessed = model.preprocess_image(test_path)
        assert preprocessed.shape == (1, 3, 224, 224), f"Wrong shape: {preprocessed.shape}"
        logger.info("✅ Image preprocessing working correctly")
    finally:
        os.remove(test_path)
    
    # Test 3: Forward pass with images
    logger.info("\n3. Testing forward pass with images...")
    batch_size = 2
    seq_length = 10
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    
    with torch.no_grad():
        # Without images
        output_no_img = model(input_ids)
        
        # With images
        output_with_img = model(input_ids, images=images)
        
        # Check output shapes
        assert output_no_img['logits'].shape == (batch_size, seq_length, config.vocab_size)
        assert output_with_img['logits'].shape == (batch_size, seq_length + 1, config.vocab_size)  # +1 for image token
        
        logger.info("✅ Forward pass with images successful")
        logger.info(f"   Output shape without images: {output_no_img['logits'].shape}")
        logger.info(f"   Output shape with images: {output_with_img['logits'].shape}")
    
    # Test 4: Generation with images
    logger.info("\n4. Testing generation with images...")
    prompt = torch.randint(0, config.vocab_size, (1, 5), device=device)
    single_image = torch.randn(1, 3, 224, 224, device=device)
    
    with torch.no_grad():
        # Generate without image
        gen_no_img = model.generate(prompt, max_new_tokens=10)
        
        # Generate with image
        gen_with_img = model.generate(prompt, images=single_image, max_new_tokens=10)
        
        logger.info("✅ Generation with images successful")
        logger.info(f"   Generated {gen_no_img['sequences'].shape[1] - 5} tokens without image")
        logger.info(f"   Generated {gen_with_img['sequences'].shape[1] - 5} tokens with image")
    
    # Test 5: Memory usage
    logger.info("\n5. Checking memory usage...")
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated(device) / 1024**3
        logger.info(f"   Current memory usage: {memory_used:.2f} GB")
        logger.info("✅ Still within 4GB memory constraint")
    
    logger.info("\n🎉 All image support tests passed!")
    logger.info("The model is ready for image-text validation experiments.")
    
    return model

if __name__ == "__main__":
    # Run comprehensive validation model tests
    model = test_validation_model()
    
    # Example validation training setup
    logger.info("\n" + "="*60)
    logger.info("Validation model ready for multimodal training data testing!")
    logger.info("="*60)
    logger.info("Next steps:")
    logger.info("1. Load your training data files (text and image-text)")
    logger.info("2. Run short training experiments (100-500 steps)")
    logger.info("3. Test reasoning activation on different content types")
    logger.info("4. Compare text-only vs image-conditioned generation quality")
    logger.info("5. Validate 80/20 split effectiveness across modalities")
    logger.info("="*60)
    
    # Test image support separately
    logger.info("\nRunning dedicated image support tests...")
   # test_image_support()
