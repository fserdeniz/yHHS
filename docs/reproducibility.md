# Reproducibility

Code and Environment
- Python 3.10+
- Dependencies in `requirements.txt`
- Recommended venv: `yHHS_env`

Steps
- Place `LTEIQ.raw` at repo root.
- Create venv and install dependencies:
  - `python3 -m venv yHHS_env`
  - `source yHHS_env/bin/activate`
  - `pip install -r requirements.txt`
- Run CLI: `python scripts/run_analysis.py LTEIQ.raw`
- Or open the notebook: `notebooks/LTE_Analiz.ipynb`

Determinism
- The pipeline is deterministic given the same input file and environment.
- Randomness is not used in detection or decoding steps.

Data Availability
- `LTEIQ.raw` is not tracked in Git. Ensure the file path matches the code/notebook expectations.
