import pandas as pd
import json
import os
from typing import Dict, List
from tqdm import tqdm
import logging

from .config import Config
from .model import ModelManager
from .segmentation import segment_abstract
from .hypothesis import generate_hypotheses
from .attribution import compute_feature_ablation, compute_shapley_values
from .evaluation import self_evaluate_hypothesis
from .utils import setup_logging, clear_memory

class ABSTRAPipeline:
    """Main ABSTRA pipeline orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.output_dir)
        self.model_manager = ModelManager(config)
    
    def process_abstract(self, row: pd.Series) -> Dict:
        """Process single abstract through all phases"""
        title = row['Title']
        abstract = row['Abstract']
        
        self.logger.info(f"Processing: {title}")
        
        result = {
            'title': title,
            'abstract': abstract,
            'hypotheses': [],
            'attribution_results': []
        }
        
        # Phase 1: Segment
        features = segment_abstract(abstract)
        result['features'] = features
        
        # Phase 2: Generate hypotheses
        hypotheses = generate_hypotheses(abstract, title, self.model_manager)
        result['hypotheses'] = hypotheses
        
        # Phase 3: Attribution & Evaluation
        for hyp_data in hypotheses:
            hyp_id = hyp_data['hypothesis_id']
            hyp_text = hyp_data['hypothesis_text']
            
            fa_scores = compute_feature_ablation(features, hyp_text, self.model_manager)
            shapley_scores = compute_shapley_values(features, hyp_text, self.model_manager, 
                                                   self.config.shapley_samples)
            eval_score, eval_text = self_evaluate_hypothesis(title, abstract, hyp_text, 
                                                             self.model_manager)
            
            result['attribution_results'].append({
                'hypothesis_id': hyp_id,
                'fa_scores': fa_scores,
                'shapley_scores': shapley_scores,
                'self_eval_score': eval_score,
                'self_eval_text': eval_text
            })
            
            clear_memory()
        
        return result
    
    def run(self) -> pd.DataFrame:
        """Execute full pipeline"""
        self.logger.info("="*50)
        self.logger.info("Starting ABSTRA Pipeline")
        self.logger.info("="*50)
        
        # Load data
        df = pd.read_csv(self.config.input_csv)
        self.logger.info(f"Loaded {len(df)} abstracts from {self.config.input_csv}")
        
        # Load model
        self.model_manager.load()
        
        # Process abstracts
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                result = self.process_abstract(row)
                results.append(result)
                
                # Reload model periodically
                if (idx + 1) % self.config.reload_model_every == 0 and idx < len(df) - 1:
                    self.logger.info("Reloading model...")
                    self.model_manager.unload()
                    self.model_manager.load()
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {str(e)}")
                continue
        
        # Save results
        self._save_results(results)
        
        # Create output DataFrame
        output_df = self._create_output_dataframe(results)
        
        self.logger.info("Pipeline complete!")
        return output_df
    
    def _save_results(self, results: List[Dict]):
        """Save JSON results"""
        json_path = os.path.join(self.config.output_dir, 'complete_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Saved JSON to {json_path}")
    
    def _create_output_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results to DataFrame"""
        rows = []
        
        for result in results:
            title = result['title']
            abstract = result['abstract']
            features = result['features']
            
            for hyp_data in result['hypotheses']:
                hyp_id = hyp_data['hypothesis_id']
                hyp_text = hyp_data['hypothesis_text']
                
                attr = next(a for a in result['attribution_results'] if a['hypothesis_id'] == hyp_id)
                
                row = {
                    'title': title,
                    'abstract': abstract,
                    'hypothesis_id': hyp_id,
                    'hypothesis': hyp_text,
                    'model_self_evaluated_score': attr['self_eval_score'],
                    'model_response': attr['self_eval_text'],
                    'abstract_background': ' '.join(features['background']),
                    'abstract_objective': ' '.join(features['objective']),
                    'abstract_methods': ' '.join(features['methods']),
                    'abstract_results': ' '.join(features['results']),
                    'abstract_conclusion': ' '.join(features['conclusion']),
                }
                
                for section, score in attr['fa_scores'].items():
                    row[f'fa_{section}'] = score
                
                for section, score in attr['shapley_scores'].items():
                    row[f'shapley_{section}'] = score
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.config.output_dir, 'abstra_results.csv')
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved CSV to {csv_path}")
        
        return df