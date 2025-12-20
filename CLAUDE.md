# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
MCM 2023 Problem C repository with three main tasks:
- Q1: Time-series forecasting for reported results
- Q2: Word attribute analysis (EERIE word analysis)
- Q3: Distribution prediction using Mixture-of-Experts (MoE) + Softmax

## Environment Setup
Create and activate the environment using:
```bash
./setup_env.sh                # Creates conda environment 'mcm2023' with Python 3.11
conda activate mcm2023
pip install -r requirements.txt
```

Alternative setup without conda:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Core Dependencies
Required packages include: numpy, pandas, matplotlib, scipy, scikit-learn, openpyxl, statsmodels, ruptures, holidays, wordfreq, nltk, tensorflow (tensorflow-macos on Apple Silicon).

## Main Entry Points and Commands

### Q1 - Time Series Forecasting
```bash
python Q1/q1_final_clean.py --input <path/to.xlsx>
# Results saved to Q1/results/*.png
```

### Q2 - Word Attribute Analysis
```bash
python 单词属性/enrich_features.py    # Feature enrichment
python 单词属性/main.py               # Main analysis
# Outputs: analysis_report.txt and heatmaps in 单词属性/
```

### Q3 - Distribution Prediction (MoE + Softmax)
```bash
python forcasting/Moe_Softmax_with_probability.py
# Results in forcasting/moe_dirichlet_output/

# Alternative newer implementations:
python task2_distribution_prediction/models/Moe_Softmax.py
# Results in moe_bootstrap_output/
```

### Quick Run Scripts
```bash
./run_task1.sh    # Run Task 1 (Q1 + Q2)
./run_task2.sh    # Run Task 2 (Q3)
```

## Directory Structure
- `Q1/` - Time-series forecasting entry point and results
- `单词属性/` - Word attribute analysis (Chinese directory name, do not rename)
- `forcasting/` - Original MoE + Softmax implementation
- `task2_distribution_prediction/` - Newer distribution prediction implementation with experiments
- `data/` - Input datasets (not tracked in git)
- `features/`, `feature_engineering/` - Feature engineering notebooks
- `models/` - Shared model code
- `archives/` - Problem statement and reference PDFs
- `pictures/` - Generated plots and visualizations
- `moe_bootstrap_output/` - MoE bootstrap results
- `shared/` - Shared utilities

## Code Architecture
- Uses mixture-of-experts (MoE) combined with Dirichlet/Softmax distributions for probabilistic forecasting
- Implements specialized change point detection using `ruptures` library
- Word analysis leverages `wordfreq` and `nltk` for linguistic features
- Time-series components use `statsmodels` and custom change point detection

## Testing and Validation
No formal test suite. Validate by running entry scripts and checking expected outputs:
- Q1: `Q1/results/*.png` files exist and contain forecast plots
- Q2: `单词属性/analysis_report.txt` and heatmap images
- Q3: `forcasting/moe_dirichlet_output/*.csv` or `moe_bootstrap_output/` results

## Data Requirements
- Place datasets in `data/` directory (not tracked)
- Excel files expected for Q1 (.xlsx format)
- Word frequency data required for Q2 (enriched via `wordfreq` package)

## Development Notes
- Maintain Chinese directory name `单词属性/` - do not rename
- Uses `pathlib` for path operations
- Follows PEP 8 with 4-space indentation, ~100 char line length
- Type hints and docstrings preferred for public functions
- Use relative imports when possible
- Seed randomness for reproducible results

## Security & Configuration
- No credentials expected in code
- Environment variables if needed for configuration
- Don't commit large datasets - keep in `data/` directory
- See `.gitignore` for excluded dataset files including glove embeddings, SUBTLEX data, and processed CSVs