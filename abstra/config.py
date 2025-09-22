import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration for ABSTRA pipeline"""
    
    # Model settings
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype: str = "float16"
    device: str = "cuda"
    
    # Paths
    input_csv: str = ""
    output_dir: str = "./results"
    
    # Processing
    num_hypotheses: int = 3
    batch_size: int = 10
    reload_model_every: int = 20
    
    # Generation
    max_length: int = 3000
    temperature: float = 0.7
    repetition_penalty: float = 1.2
    
    # Attribution
    shapley_samples: int = 10
    
    # Evaluation
    eval_max_length: int = 1024
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model_name=config_dict['model']['name'],
            dtype=config_dict['model']['dtype'],
            device=config_dict['model']['device'],
            input_csv=config_dict['paths']['input_csv'],
            output_dir=config_dict['paths']['output_dir'],
            num_hypotheses=config_dict['processing']['num_hypotheses'],
            batch_size=config_dict['processing']['batch_size'],
            reload_model_every=config_dict['processing']['reload_model_every'],
            max_length=config_dict['generation']['max_length'],
            temperature=config_dict['generation']['temperature'],
            repetition_penalty=config_dict['generation']['repetition_penalty'],
            shapley_samples=config_dict['attribution']['shapley_samples'],
            eval_max_length=config_dict['evaluation']['max_length']
        )