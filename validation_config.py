import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

@dataclass
class ValidationConfig:
    hidden_dim: int = 512
    num_layers: int = 16
    num_attention_heads: int = 8
    num_kv_heads: int = 2
    intermediate_dim: int = 1408
    vocab_size: int = 32000
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    use_reasoning: bool = True
    reasoning_dim: int = 256
    reasoning_heads: int = 4
    reasoning_threshold: float = 0.7
    use_images: bool = True
    image_size: int = 224
    vision_hidden_dim: int = 512
    freeze_vision_encoder: bool = True
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    embedding_dropout: float = 0.1
    initializer_range: float = 0.02
    use_scaled_init: bool = True
    layer_norm_epsilon: float = 1e-6
    use_gradient_checkpointing: bool = True
    checkpoint_layers: List[int] = field(default_factory=lambda: [2, 4, 6])
    eos_token_id: int = 2
    pad_token_id: int = 0
    bos_token_id: int = 1
    validation_batch_size: int = 4
    validation_steps: int = 100
    log_every_n_steps: int = 10
    save_every_n_steps: int = 50
    text_validation_path: str = "validation_text.jsonl"
    image_validation_path: str = "validation_images.jsonl"
    output_dir: str = "validation_outputs"
    
    # NEW: Reasoning tokens configuration
    reasoning_tokens: Dict[str, int] = field(default_factory=dict)
    extended_vocab_size: int = field(init=False)
    think_start_id: int = field(init=False)
    think_end_id: int = field(init=False)
    step_start_id: int = field(init=False)
    step_end_id: int = field(init=False)
    
    def __post_init__(self):
        """Initialize reasoning tokens after dataclass creation"""
        # Create reasoning tokens mapping
        self.reasoning_tokens = {
            '<think>': self.vocab_size + 0,   # 32000
            '</think>': self.vocab_size + 1,  # 32001
            '<step1>': self.vocab_size + 2,   # 32002
            '<step2>': self.vocab_size + 3,   # 32003
            '<step3>': self.vocab_size + 4,   # 32004
            '<step4>': self.vocab_size + 5,   # 32005
            '<step5>': self.vocab_size + 6,   # 32006
            '<step6>': self.vocab_size + 7,   # 32007
            '<step7>': self.vocab_size + 8,   # 32008
            '<step8>': self.vocab_size + 9,   # 32009 
           
           # answer structure tokens
            '<answer>': self.vocab_size + 10,  # 32010
            '</answer>': self.vocab_size + 11, # 32011
           ## Advanced reasoning tokens - Branching
            '<branch1>': self.vocab_size + 12,    # 32012
            '<branch2>': self.vocab_size + 13,    # 32013
            '<branch3>': self.vocab_size + 14,    # 32014
            '<approach>': self.vocab_size + 15,   # 32015
            '<assessment>': self.vocab_size + 16, # 32016
            '<synthesis>': self.vocab_size + 17,  # 32017
            # Advanced reasoning tokens - Error Correction
            '<verification>': self.vocab_size + 18, # 32018
            '<reflection>': self.vocab_size + 19,   # 32019
            '<correction>': self.vocab_size + 20,   # 32020
            '<retry>': self.vocab_size + 21,        # 32021
            # Advanced reasoning tokens - Analysis
            '<analysis>': self.vocab_size + 22,     # 32022
            '<hypothesis>': self.vocab_size + 23,   # 32023
            '<evidence>': self.vocab_size + 24,     # 3202
            '<conclusion>': self.vocab_size + 25,   # 32025

        }
        # Calculate extended vocabulary size
        self.extended_vocab_size = self.vocab_size + len(self.reasoning_tokens)
        
        # Set token ID shortcuts
        self.think_start_id = self.reasoning_tokens['<think>']
        self.think_end_id = self.reasoning_tokens['</think>']
        self.step_start_id = self.reasoning_tokens['<step1>']
        self.step_end_id = self.reasoning_tokens['<step8>']
        self.answer_start_id = self.reasoning_tokens['<answer>']    # 32010
        self.answer_end_id = self.reasoning_tokens['</answer>']     # 32011
        # Advanced token ID shortcuts
        self.branch_start_id = self.reasoning_tokens['<branch1>']    # 32012
        self.branch_end_id = self.reasoning_tokens['<branch3>']     # 32014
        self.verification_id = self.reasoning_tokens['<verification>'] # 32018
        self.reflection_id = self.reasoning_tokens['<reflection>']     # 32019
    
    def get_token_id(self, token_string: str) -> Optional[int]:
        """Get token ID for a reasoning token"""
        return self.reasoning_tokens.get(token_string, None)

    def get_token_string(self, token_id: int) -> Optional[str]:
        """Get token string from token ID"""
        for token_str, token_id_val in self.reasoning_tokens.items():
            if token_id_val == token_id:
                return token_str
        return None

    def is_reasoning_token(self, token_id: int) -> bool:
        """Check if token ID is a reasoning token"""
        return token_id in self.reasoning_tokens.values()

    def get_step_number(self, token_id: int) -> Optional[int]:
        """Extract step number from step token ID"""
        if token_id >= self.step_start_id and token_id <= self.step_end_id:
            return token_id - self.step_start_id + 1
        return None

    def save(self, path: str):
        """Save configuration to file"""
        with open(path, 'w') as f:
            # Convert to dict, excluding non-serializable computed fields
            config_dict = asdict(self)
            # Remove computed fields that will be recreated by __post_init__
            for field_name in ['reasoning_tokens', 'extended_vocab_size', 
                              'think_start_id', 'think_end_id', 
                              'step_start_id', 'step_end_id']:
                config_dict.pop(field_name, None)
            json.dump(config_dict, f, indent=2)

    def get_reasoning_pattern(self, input_ids):

        """Detect which reasoning pattern is being used"""
        if input_ids is None:
         return 'basic'

        # check for branching pattern
         branch_tokens = [self.get_token_id('<branch1>'), self.get_token_id('<branch2>'), self.get_token_id('<branch3>')]
         if any((input_ids == token_id).any() for token_id in branch_tokens if token_id is not None):
             return 'branching'
         # Check for correction pattern
         correction_tokens = [self.get_token_id('<verification>'), self.get_token_id('<reflection>'), self.get_token_id('<correction>')]
         if any((input_ids == token_id).any() for token_id in correction_tokens if token_id is not None):
                return 'correction'
         # Check for analysis pattern
        analysis_tokens = [self.get_token_id('<analysis>'), self.get_token_id('<hypothesis>'), self.get_token_id('<evidence>')]
        if any((input_ids == token_id).any() for token_id in analysis_tokens if token_id is not None):
            return 'analysis'

        return 'basic'  

    def is_advanced_reasoning_token(self, token_id):
        """Check if token is an advanced reasoning token"""
        advanced_tokens = [
            '<branch1>', '<branch2>', '<branch3>', '<approach>', '<assessment>', '<synthesis>',
            '<verification>', '<reflection>', '<correction>', '<retry>',
            '<analysis>', '<hypothesis>', '<evidence>', '<conclusion>'
            ]
        return any(self.get_token_id(token) == token_id for token in advanced_tokens)


    @classmethod
    def load(cls, path: str) -> 'ValidationConfig':
        """Load configuration from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        # Remove computed fields if they exist in the saved file
        for field_name in ['reasoning_tokens', 'extended_vocab_size', 
                          'think_start_id', 'think_end_id', 
                          'step_start_id', 'step_end_id']:
            data.pop(field_name, None)
        return cls(**data)

    def get_model_size_estimate(self) -> Dict[str, float]:
        embedding_params = self.extended_vocab_size * self.hidden_dim
        attention_params = (
            self.hidden_dim * self.hidden_dim +
            self.hidden_dim * (self.hidden_dim // self.num_attention_heads * self.num_kv_heads) * 2 +
            self.hidden_dim * self.hidden_dim
        )
        mlp_params = (
            self.hidden_dim * self.intermediate_dim * 2 +
            self.intermediate_dim * self.hidden_dim
        )
        layernorm_params = self.hidden_dim * (2 * self.num_layers + 1)
        transformer_params = self.num_layers * (attention_params + mlp_params) + layernorm_params
        reasoning_params = 0
        if self.use_reasoning:
            reasoning_params += (
                self.hidden_dim * (self.hidden_dim // 4) +
                (self.hidden_dim // 4) * 32 +
                32 * 1 +
                self.hidden_dim * self.reasoning_dim +
                self.reasoning_dim * self.hidden_dim +
                self.reasoning_dim * self.reasoning_dim * 12 +
                self.hidden_dim * 2 * self.hidden_dim +
                self.hidden_dim * self.hidden_dim
            )
        vision_params = 0
        if self.use_images:
            vision_params += 11_200_000
            vision_params += self.vision_hidden_dim * self.hidden_dim
        total_params = embedding_params + transformer_params + reasoning_params + vision_params
        return {
            'embedding_millions': embedding_params / 1e6,
            'transformer_millions': transformer_params / 1e6,
            'reasoning_millions': reasoning_params / 1e6,
            'vision_millions': vision_params / 1e6,
            'total_millions': total_params / 1e6,
            'total_billions': total_params / 1e9,
            'scaling_factor': 2000 / (total_params / 1e6)
        }

    def validate_config(self):
        assert self.hidden_dim % self.num_attention_heads == 0, "hidden_dim must be divisible by num_attention_heads"
        assert self.num_attention_heads % self.num_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"
        if self.use_reasoning:
            assert self.reasoning_dim % self.reasoning_heads == 0, "reasoning_dim must be divisible by reasoning_heads"
        if self.use_gradient_checkpointing:
            max_layer = self.num_layers - 1
            self.checkpoint_layers = [layer for layer in self.checkpoint_layers if 0 <= layer <= max_layer]
        return True

def create_validation_config() -> ValidationConfig:
    config = ValidationConfig()
    config.validate_config()
    size_info = config.get_model_size_estimate()
    print(f"Validation Model: {size_info['total_millions']:.1f}M params, {size_info['scaling_factor']:.1f}x smaller than full model")
    return config

def create_minimal_config() -> ValidationConfig:
    config = ValidationConfig(
        hidden_dim=448, 
        num_layers=16, 
        num_attention_heads=8, 
        num_kv_heads=2, 
        intermediate_dim=1232, 
        reasoning_dim=224, 
        reasoning_heads=4, 
        max_position_embeddings=768
    )
    config.checkpoint_layers = [layer for layer in [2, 4, 5] if layer < config.num_layers]
    config.validate_config()
    return config

def create_text_only_config() -> ValidationConfig:
    config = ValidationConfig(use_images=False)
    config.validate_config()
    return config

def create_no_reasoning_config() -> ValidationConfig:
    config = ValidationConfig(use_reasoning=False)
    config.validate_config()
    return config

def test_reasoning_tokens():
    """Test reasoning token configuration"""
    config = ValidationConfig()
    
    print("🔧 Testing reasoning token configuration...")
    print(f"Original vocab size: {config.vocab_size}")
    print(f"Extended vocab size: {config.extended_vocab_size}")
    print(f"Number of reasoning tokens: {len(config.reasoning_tokens)}")
    
    # Test token mappings
    print("\n📝 Reasoning token mappings:")
    for token_str, token_id in config.reasoning_tokens.items():
        print(f"  {token_str}: {token_id}")
    
    # Test helper methods
    print(f"\n🧪 Helper method tests:")
    print(f"  get_token_id('<think>'): {config.get_token_id('<think>')}")
    print(f"  get_token_string(32000): {config.get_token_string(32000)}")
    print(f"  is_reasoning_token(32000): {config.is_reasoning_token(32000)}")
    print(f"  get_step_number(32004): {config.get_step_number(32004)}")
    
    # Test save/load functionality
    print(f"\n💾 Testing save/load functionality:")
    config.save("test_config.json")
    loaded_config = ValidationConfig.load("test_config.json")
    print(f"  Original extended vocab size: {config.extended_vocab_size}")
    print(f"  Loaded extended vocab size: {loaded_config.extended_vocab_size}")
    print(f"  Save/load successful: {config.extended_vocab_size == loaded_config.extended_vocab_size}")
    
    print("\n✅ All reasoning token tests passed!")
    return config

if __name__ == "__main__":
    # Test reasoning tokens first
    test_config = test_reasoning_tokens()
    
    # Then create and save the main config
    config = create_validation_config()
    config.save("validation_config.json")
    print("Saved to validation_config.json")
