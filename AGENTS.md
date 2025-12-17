# Repository Guidelines

## Project Structure & Module Organization
- `Q1/` — Time-series forecasting for reported results. Entry: `Q1/q1_final_clean.py`. Outputs in `Q1/results/`.
- `单词属性/` — Word attribute analysis (Hard Mode, difficulty). Entry: `单词属性/main.py`. Writes `analysis_report.txt` and heatmaps here.
- `forcasting/` — Distribution prediction (Mixture-of-Experts + Softmax). Entry: `forcasting/Moe_Softmax.py`. Outputs in `forcasting/moe_output/`.
- `features/` — Feature engineering notebooks and generators used to build `data/mcm_processed_data.csv`.
- `data/` — Input datasets (not tracked). Place local data here.
- `archives/` — Problem statement and reference PDFs.
- `pictures/`, `backups/`, `util/`, `models/` — Assets, intermediate artifacts, helpers, and model code.

## Build, Test, and Development Commands
- Create env and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - Optional extras used in docs/scripts: `pip install ruptures holidays statsmodels seaborn xgboost nltk wordfreq torch`
- Run main flows:
  - Q1: `python Q1/q1_final_clean.py --input <path/to.xlsx>` → results in `Q1/results/`
  - Q2: `python 单词属性/enrich_features.py` then `python 单词属性/main.py`
  - Q3: `python forcasting/Moe_Softmax.py` (see outputs in `forcasting/moe_output/`)

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation, max line length ~100.
- Naming: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_CASE` for constants.
- Prefer type hints and docstrings for public functions. Use `pathlib` and relative paths.
- Keep existing directory names as-is (e.g., `单词属性/`, `forcasting/`).

## Testing Guidelines
- No formal test suite. Use smoke runs of each entry script and verify expected artifacts exist (e.g., `Q1/results/*.png`, `analysis_report.txt`, `moe_output/*.csv`).
- Seed randomness where applicable and avoid hard-coded absolute paths.

## Commit & Pull Request Guidelines
- Commits: concise, imperative. Optional type/scope: `feat`, `fix`, `refactor`, `perf`, `docs` (e.g., `perf(Q3): improve training step time`).
- PRs: include a clear description, reproduction commands, and sample outputs/paths. Link issues when relevant. Add screenshots from `Q1/results/` or `forcasting/moe_output/` when visual changes occur.
- Do not commit large datasets or bulky artifacts; keep them in `data/` or the respective output folders and update `.gitignore` if needed.

## Security & Configuration Tips
- Keep credentials out of code and config. Use environment variables if needed.
- Document any new data dependencies in `README.md` and prefer small, versioned sample files for examples.

## Agent-Specific Notes
- Avoid renaming/moving top-level directories; changes can break imports and paths.
- Make minimal, focused patches aligned with this guide and the existing structure.
