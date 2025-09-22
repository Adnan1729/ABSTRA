import torch
import gc
import logging
import os

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'abstra_log.txt')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_path}")
    return logger

def clear_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()