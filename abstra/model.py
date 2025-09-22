import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import clear_memory
from .config import Config
import logging

class ModelManager:
    """Manages model loading and text generation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
    
    def load(self):
        """Load model and tokenizer"""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        dtype = torch.float16 if self.config.dtype == "float16" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        
        self.logger.info("Model loaded successfully")
    
    def generate(self, prompt: str, max_length: int = None) -> str:
        """Generate text from prompt"""
        if max_length is None:
            max_length = self.config.max_length
        
        try:
            messages = [{"role": "user", "content": prompt}]
            chat = self.tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = self.tokenizer(chat, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=self.config.temperature,
                    do_sample=True,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            clear_memory()
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return ""
    
    def unload(self):
        """Unload model from memory"""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        clear_memory()