import nltk
from typing import Dict, List

nltk.download('punkt', quiet=True)

def segment_abstract(abstract: str) -> Dict[str, List[str]]:
    """
    Segment abstract into five sections using position-based approach
    
    Args:
        abstract: Full abstract text
        
    Returns:
        Dictionary with keys: background, objective, methods, results, conclusion
    """
    try:
        sentences = nltk.sent_tokenize(abstract)
    except:
        sentences = [s.strip() for s in abstract.split('.') if s.strip()]
    
    features = {
        'background': [],
        'objective': [],
        'methods': [],
        'results': [],
        'conclusion': []
    }
    
    if len(sentences) <= 3:
        features['methods'] = sentences
    else:
        for i, sentence in enumerate(sentences):
            position = i / len(sentences)
            if position < 0.2:
                features['background'].append(sentence)
            elif position < 0.4:
                features['objective'].append(sentence)
            elif position < 0.6:
                features['methods'].append(sentence)
            elif position < 0.8:
                features['results'].append(sentence)
            else:
                features['conclusion'].append(sentence)
    
    return features