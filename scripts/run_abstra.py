#!/usr/bin/env python
"""Main script to run ABSTRA pipeline"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from abstra import ABSTRAPipeline, Config

def main():
    parser = argparse.ArgumentParser(description="Run ABSTRA pipeline")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--input', type=str, help='Input CSV path (overrides config)')
    parser.add_argument('--output', type=str, help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Override with command line args
    if args.input:
        config.input_csv = args.input
    if args.output:
        config.output_dir = args.output
    
    # Run pipeline
    pipeline = ABSTRAPipeline(config)
    results_df = pipeline.run()
    
    print("\n" + "="*50)
    print("SUCCESS! Pipeline completed")
    print(f"Output shape: {results_df.shape}")
    print("="*50)

if __name__ == "__main__":
    main()