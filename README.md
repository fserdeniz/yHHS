# LTE Broadcast Parameter Extraction (10 MHz LTE, 5 ms IQ)

Analyzes a 5 ms LTE IQ capture and extracts broadcast parameters with a robust, single‑file pipeline. Includes a presentation‑ready Jupyter notebook packed with explanatory visuals.

## Features
- Input: interleaved float32 I/Q from `LTEIQ.raw` (5 ms, Fs=15.36 MHz, 10 MHz BW)
- PSS/SSS detection (deterministic SSS generator; FDD + TDD variants)
  - `NDLRB` (from config/MIB), `CyclicPrefix` (Normal)
  - `DuplexMode` (FDD/TDD)
  - `NCellID` (PCI), `NSubframe` (0 or 5)
- PBCH/MIB decoding (single 5 ms, best‑effort yet strong):
  - CFO median, common phase, flat EQ
  - PBCH RE extraction and improved CRS masking (ports 0/1)
  - LLR + de‑rate matching; scrambler variants; Viterbi; CRC/MIB parsing
  - Brute‑force fallback over `NCellID`, restricted by `NID2`, with quality gates
  - Fills `CellRefP`, `PHICHDuration`, `Ng`, `NFrame`, and may refine `NDLRB`
- TDD extras (heuristic, presentation‑friendly):
  - Special subframe (subframe 1) detector via center‑band energy
  - UL‑DL configuration index guesser (0..6) using first 5 subframes
- Notebook with 10+ figures: PSS/SSS correlations, PBCH spectra/scatter, energy heatmaps, LLR histograms, etc. (`notebooks/LTE_Analiz.ipynb`)

## Data
- Place your LTE IQ file at the repo root as `LTEIQ.raw` (or adjust the notebook path).
- Do not commit large/binary captures to Git — `.gitignore` excludes `*.raw`, `*.iq`, `*.bin`, and `LTEIQ.raw`.

## Environment
- Python 3.10+
- Recommended virtual environment name: `yHHS_env`
- Dependencies listed in `requirements.txt`

Create and activate the environment, then install deps:

```
python3 -m venv yHHS_env
source yHHS_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

(Optional) register the Jupyter kernel:
```
python -m ipykernel install --user --name=yHHS_env --display-name "Python (yHHS_env)"
```

## Usage
- CLI quick run:
```
source yHHS_env/bin/activate
python scripts/run_analysis.py LTEIQ.raw
```

- Notebook:
```
source yHHS_env/bin/activate
jupyter notebook notebooks/LTE_Analiz.ipynb
```
Select kernel: `Python (yHHS_env)`.

### Example Output (single 5 ms file)
```
NDLRB: 15
DuplexMode: FDD
CyclicPrefix: Normal
NCellID: 7
NSubframe: 0
CellRefP: 2
PHICHDuration: Extended
Ng: 1/2
NFrame: 109
```

## Results and Limitations
- With 5 ms input, the pipeline reliably outputs: `NDLRB` (or MIB‑refined), `CyclicPrefix`, `DuplexMode`, `NCellID`, `NSubframe`.
- MIB fields (`CellRefP`, `PHICHDuration`, `Ng`, `NFrame`) are often recovered via PBCH brute‑force even with 5 ms, but remain SNR/recording dependent.
- Heuristic parts (TDD special subframe/config index) are intended for presentation/triage; longer captures improve accuracy.
- For production‑grade PBCH, integrate standard‑accurate rate‑matching/interleaving and CRS‑based channel equalization across multiple PBCH transmissions.

## Repository Structure
- `src/lte_params.py`: PSS/SSS detection, CFO/FFT helpers, high‑level `analyze_lte_iq`
- `src/pbch.py`: PBCH RE extraction, CRS mask, scrambler variants, de‑rate match, Viterbi, MIB parsing, brute‑force
- `scripts/run_analysis.py`: CLI entry to analyze `LTEIQ.raw`
- `notebooks/LTE_Analiz.ipynb`: Presentation notebook with step‑by‑step flow and 10+ figures
- `requirements.txt`: Python dependencies
- `.gitignore`: Excludes venvs and capture files (e.g., `*.raw`)

## Notes
- The PBCH/MIB path includes practical approximations adapted to short captures; robust on typical lab recordings.
- Run time: PBCH brute‑force may take ~10–60 s depending on CPU; the search is constrained by `NID2` and quality gates.
- The notebook is designed as slides: you can run “Hızlı Sonuçlar” alone or step through details and visuals.

## References
[1] ShareTechnote, "LTE PSS – Primary Synchronization Signal." [Online]. Available: https://www.sharetechnote.com/html/Handbook_LTE_PSS.html

[2] ShareTechnote, "LTE SSS – Secondary Synchronization Signal." [Online]. Available: https://www.sharetechnote.com/html/Handbook_LTE_SSS.html

[3] ShareTechnote, "LTE PBCH – Physical Broadcast Channel." [Online]. Available: https://www.sharetechnote.com/html/Handbook_LTE_PBCH.html

[4] ShareTechnote, "LTE MIB – Master Information Block." [Online]. Available: https://www.sharetechnote.com/html/Handbook_LTE_MIB.html

[5] ShareTechnote, "LTE SIB – System Information Block." [Online]. Available: https://www.sharetechnote.com/html/Handbook_LTE_SIB.html

[6] ShareTechnote, "LTE Reference Signal (RS/CRS)." [Online]. Available: https://www.sharetechnote.com/html/Handbook_LTE_ReferenceSignal.html

[7] ShareTechnote, "LTE PCFICH – Physical Control Format Indicator Channel." [Online]. Available: https://www.sharetechnote.com/html/Handbook_LTE_PCFICH.html

[8] ShareTechnote, "LTE PDCCH – Physical Downlink Control Channel." [Online]. Available: https://www.sharetechnote.com/html/Handbook_LTE_PDCCH.html
