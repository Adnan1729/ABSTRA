import torch
from typing import Dict, List
from captum.attr import FeatureAblation, ShapleyValueSampling, LLMAttribution, TextTemplateInput
from .model import ModelManager
from .utils import clear_memory
import logging

def compute_feature_ablation(features: Dict[str, List[str]], hypothesis: str, 
                            model_manager: ModelManager) -> Dict[str, float]:
    """Compute Feature Ablation attribution scores"""
    logger = logging.getLogger(__name__)
    
    all_sentences = []
    feature_indices = {}
    current_idx = 0
    
    for section in ['background', 'objective', 'methods', 'results', 'conclusion']:
        feature_indices[section] = (current_idx, current_idx + len(features[section]))
        all_sentences.extend(features[section])
        current_idx += len(features[section])
    
    if not all_sentences:
        return {s: 0.0 for s in ['background', 'objective', 'methods', 'results', 'conclusion']}
    
    try:
        template = " ".join(["{}"]*len(all_sentences))
        inp = TextTemplateInput(template, values=all_sentences)
        
        fa = FeatureAblation(model_manager.model)
        llm_attr = LLMAttribution(fa, model_manager.tokenizer)
        
        with torch.amp.autocast('cuda'):
            attr_res = llm_attr.attribute(inp, target=hypothesis, skip_tokens=torch.tensor([1]))
        
        section_scores = {}
        for section, (start, end) in feature_indices.items():
            section_scores[section] = attr_res.seq_attr[start:end].mean().item() if end > start else 0.0
        
        clear_memory()
        return section_scores
    except Exception as e:
        logger.error(f"FA error: {str(e)}")
        return {s: 0.0 for s in ['background', 'objective', 'methods', 'results', 'conclusion']}

def compute_shapley_values(features: Dict[str, List[str]], hypothesis: str, 
                          model_manager: ModelManager, n_samples: int = 10) -> Dict[str, float]:
    """Compute Shapley Value attribution scores"""
    logger = logging.getLogger(__name__)
    
    all_sentences = []
    feature_indices = {}
    current_idx = 0
    
    for section in ['background', 'objective', 'methods', 'results', 'conclusion']:
        feature_indices[section] = (current_idx, current_idx + len(features[section]))
        all_sentences.extend(features[section])
        current_idx += len(features[section])
    
    if not all_sentences:
        return {s: 0.0 for s in ['background', 'objective', 'methods', 'results', 'conclusion']}
    
    try:
        template = " ".join(["{}"]*len(all_sentences))
        inp = TextTemplateInput(template, values=all_sentences)
        
        shapley = ShapleyValueSampling(model_manager.model)
        llm_attr = LLMAttribution(shapley, model_manager.tokenizer)
        
        with torch.amp.autocast('cuda'):
            attr_res = llm_attr.attribute(inp, target=hypothesis, n_samples=n_samples)
        
        section_scores = {}
        for section, (start, end) in feature_indices.items():
            section_scores[section] = attr_res.seq_attr[start:end].mean().item() if end > start else 0.0
        
        clear_memory()
        return section_scores
    except Exception as e:
        logger.error(f"Shapley error: {str(e)}")
        return {s: 0.0 for s in ['background', 'objective', 'methods', 'results', 'conclusion']}