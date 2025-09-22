import re
from typing import Tuple
from .model import ModelManager

def self_evaluate_hypothesis(title: str, abstract: str, hypothesis: str, 
                            model_manager: ModelManager) -> Tuple[float, str]:
    """
    Self-evaluate hypothesis quality
    
    Returns:
        Tuple of (score, explanation_text)
    """
    prompt = f"""Evaluate this scientific hypothesis against the abstract.

Title: {title}
Abstract: {abstract}
Hypothesis: {hypothesis}

Rate from 0.0 to 1.0 based on how well it represents the abstract.
Provide your final score as "FINAL SCORE: X.X"
"""
    
    response = model_manager.generate(prompt, max_length=1024)
    
    # Extract score
    match = re.search(r"FINAL SCORE:\s*(\d+\.?\d*)", response, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        return round(min(max(score, 0), 1), 1), response
    
    # Fallback
    matches = re.findall(r"(\d+\.\d+)", response)
    for m in matches:
        num = float(m)
        if 0 <= num <= 1:
            return round(num, 1), response
    
    return 0.5, response