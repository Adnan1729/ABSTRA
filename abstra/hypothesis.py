from typing import List, Dict
from .model import ModelManager

def generate_hypotheses(abstract: str, title: str, model_manager: ModelManager) -> List[Dict]:
    """
    Generate multiple hypotheses for an abstract
    
    Args:
        abstract: Abstract text
        title: Paper title
        model_manager: ModelManager instance
        
    Returns:
        List of hypothesis dictionaries
    """
    prompts = [
        f"Read this scientific paper abstract and identify its main hypothesis.\n\nTitle: {title}\nAbstract: {abstract}\n\nWhat is the main hypothesis?",
        f"Based on this abstract titled '{title}', what is the central hypothesis being tested?\n\nAbstract: {abstract}",
        f"Scientific Abstract: {abstract}\nTitle: {title}\n\nExtract the primary research hypothesis. Be specific."
    ]
    
    hypotheses = []
    for i, prompt in enumerate(prompts):
        hyp_text = model_manager.generate(prompt)
        
        if "<|assistant|>" in hyp_text:
            hyp_text = hyp_text.split("<|assistant|>")[1].strip()
        
        hypotheses.append({
            'hypothesis_id': i + 1,
            'hypothesis_text': hyp_text
        })
    
    return hypotheses