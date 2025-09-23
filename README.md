# ABSTRA: Abstract Section-Targeted Reasoning Assessment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A research framework for analyzing how Large Language Models generate scientific hypotheses from research paper abstracts through multi-method attribution analysis.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Input/Output Specifications](#inputoutput-specifications)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)
- [Correspondence](#correspondence)

## Overview

ABSTRA aims to systematise our understanding of how LLMs process scientific abstracts and generate hypotheses, with particular focus on identifying implausible or extreme hypotheses that might impede automated scientific discovery processes. The ABSTRA framework transforms our empirical findings into a practical tool that addresses a critical gap in scientific AI systems: distinguishing promising hypotheses from implausible ones. By operationalising our core finding that LLMs prioritise empirical information over interpretive synthesis, it enables the generation of empirically-grounded hypotheses while filtering out those that extrapolate beyond available evidence.

## Installation

### Setup

```bash
# Clone the repository
git clone https://github.com/Adnan1729/abstra.git
cd abstra

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Quick Start

```bash
python scripts/run_abstra.py --input data/sample/sample_data_2.csv --output data/processed/
```

## Project Structure

```
abstra/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation config
├── config.yaml               # Pipeline configuration
├── .gitignore                # Git ignore rules
│
├── abstra/                   # Main package
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration management
│   ├── model.py              # LLM loading and generation
│   ├── segmentation.py       # Abstract segmentation
│   ├── hypothesis.py         # Hypothesis generation
│   ├── attribution.py        # Feature Ablation & Shapley
│   ├── evaluation.py         # Self-evaluation
│   ├── pipeline.py           # Main orchestrator
│   └── utils.py              # Logging and utilities
│
├── scripts/
│   └── run_abstra.py         # CLI entry point
│
├── data/
│   ├── README.md             # Data documentation
│   ├── raw/                  # Input CSV files (gitignored)
│   ├── processed/            # Output files (gitignored)
│   └── sample/               # Sample data (committed)
│       └── sample_data_2.csv
│
├── notebooks/                # Jupyter notebooks (optional)
│   └── example_usage.ipynb
│
└── tests/                    # Unit tests
    └── __init__.py
```

## Pipeline Architecture

### Phase 1: Literature Processing
```
Input: Abstract text
↓
Position-based sentence segmentation
↓
Output: 5 sections (Background, Objective, Methods, Results, Conclusion)
```

### Phase 2: Hypothesis Generation
```
Input: Abstract + Title
↓
3 different prompts to LLM
↓
Output: 3 hypotheses per abstract
```

### Phase 3: Attribution Analysis
```
Input: Segmented abstract + Hypothesis
↓
Feature Ablation (measures independent contribution)
Shapley Values (captures feature interactions)
↓
Output: Attribution scores per section
```

### Phase 4: Self-Evaluation
```
Input: Abstract + Hypothesis
↓
LLM evaluates hypothesis quality (0.0-1.0)
↓
Output: Score + explanation
```

### Phase 5: Output Generation
```
All results compiled into:
- CSV (abstra_results.csv)
- JSON (complete_results.json)
- Logs (abstra_log.txt)
```

## Input/Output Specifications

### Input Format

**CSV file with 2 columns:**

| Column | Type | Description |
|--------|------|-------------|
| `Title` | String | Paper title |
| `Abstract` | String | Full abstract text |

**Example:**
```csv
Title,Abstract
"Deep Learning for Medical Diagnosis","Background: Medical diagnosis requires extensive expertise. Objective: We develop a deep learning system. Methods: CNN architecture on 10,000 images. Results: 95% accuracy achieved. Conclusion: AI shows promise in healthcare."
```

### Output Format

**CSV with 21 columns:**

| Column Group | Columns | Description |
|--------------|---------|-------------|
| **Basic Info** | `title`, `abstract`, `hypothesis_id`, `hypothesis` | Paper and hypothesis details |
| **Evaluation** | `model_self_evaluated_score`, `model_response` | LLM self-assessment (0.0-1.0) |
| **Segmentation** | `abstract_background`, `abstract_objective`, `abstract_methods`, `abstract_results`, `abstract_conclusion` | Segmented abstract sections |
| **Feature Ablation** | `fa_background`, `fa_objective`, `fa_methods`, `fa_results`, `fa_conclusion` | Attribution scores |
| **Shapley Values** | `shapley_background`, `shapley_objective`, `shapley_methods`, `shapley_results`, `shapley_conclusion` | Attribution scores |

**Sample Output Row:**
```csv
title,abstract,hypothesis_id,hypothesis,model_self_evaluated_score,model_response,abstract_background,abstract_objective,abstract_methods,abstract_results,abstract_conclusion,fa_background,fa_objective,fa_methods,fa_results,fa_conclusion,shapley_background,shapley_objective,shapley_methods,shapley_results,shapley_conclusion
"Medical AI Study","Full abstract text...",1,"The hypothesis that...",0.8,"FINAL SCORE: 0.8 because...","Background text","Objective text","Methods text","Results text","Conclusion text",0.15,0.22,0.18,0.35,0.10,0.12,0.25,0.20,0.33,0.10
```

### Output Files

1. **`abstra_results.csv`** - Main results table (21 columns × 3N rows, where N = number of abstracts)
2. **`complete_results.json`** - Complete pipeline output with nested structure
3. **`abstra_log.txt`** - Execution logs with timestamps

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Hugging Face model
  dtype: "float16"                             # Precision
  device: "cuda"                               # Device

paths:
  input_csv: "./data/sample/sample_data_2.csv"
  output_dir: "./data/processed"

processing:
  num_hypotheses: 3        # Hypotheses per abstract
  batch_size: 10           # Processing batch size
  reload_model_every: 20   # Memory management

generation:
  max_length: 3000         # Max tokens for hypothesis
  temperature: 0.7         # Sampling temperature
  repetition_penalty: 1.2  # Reduce repetition

attribution:
  shapley_samples: 10      # Shapley sampling iterations

evaluation:
  max_length: 1024         # Max tokens for evaluation
```

## Citation

If you use ABSTRA in your research, please cite:

```bibtex
@misc{abstra2025,
  title        = {ABSTRA: An Empirical Framework for Evaluating LLM-Generated Scientific Hypotheses},
  author       = {Adnan Mahmud and Abbi Abdel Rehim and Gabriel Reader and Ross King},
  year         = {2025},
  howpublished = {Preprint},
  url          = {https://github.com/Adnan1729/abstra}
}
```

## License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the Apache License 2.0.  
You are free to use, modify, and distribute this work, provided you comply with the terms of the license.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Captum](https://captum.ai/) for attribution methods
- [TinyLlama Team](https://github.com/jzhang38/TinyLlama) for the base model

## Correspondence

- **Point of Contact:** Adnan Mahmud
- **Email**: mam255@cantab.ac.uk
- **Project Link**: [https://github.com/Adnan1729/abstra](https://github.com/Adnan1729/abstra)

---

**Note**: This research tool is a proof-of-concept currently undergoing validation and refinement.
