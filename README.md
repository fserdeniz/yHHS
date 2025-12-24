# LTE Broadcast Parameter Extraction (Auto LTE Bandwidth, 5 ms IQ)

Analyzes a short LTE IQ capture and extracts broadcast parameters with a robust, single‑file pipeline. Includes a presentation‑ready Jupyter notebook packed with explanatory visuals.

- Input: interleaved float32 I/Q (`.raw`/`.iq`) or `.mat`; auto-selects LTE bandwidth (1.4–20 MHz, Normal CP) based on the capture. A sampling-rate hint in `.mat` (`fs`, `Fs`, `samp_rate`, etc.) is used when available.
- PSS/SSS detection:
  - PSS correlation uses a matched-filter search comparable to standard LTE cell-search implementations, providing robust timing, CFO, and frame-align offsets for all NID2 ∈ {0,1,2}
  - **SSS generation now follows 3GPP TS 36.211 § 6.11.2 exactly** (deterministic x_s/x_c/x_z recursions, q/q′ → m₀/m₁, even/odd mapping)
  - Auto-detects PCI directly from the capture (no hard-coded calibration), reporting the best-match `(NID1, NID2)` / PCI and SSS-derived subframe/duplex hypotheses
  - Outputs `NDLRB` (config/MIB), `CyclicPrefix`, `DuplexMode` (FDD/TDD), `NCellID` (PCI), `NSubframe`
- PBCH/MIB decoding (single 5 ms, work‑in‑progress toward full compliance):
  - Spec Gold descrambler (TS 36.211 § 6.6.1)
  - Sub-block interleaver + circular buffer rate matching (TS 36.212 § 5.1.4.2) and tail-biting Viterbi (rate‑1/3, K=7)
  - Brute-force fallback over `NCellID`, filtered by `NID2`
  - Optional CRS-based PBCH equalisation candidate (TS 36.211 § 6.10.1); still best-effort on short captures, MIB fields are not guaranteed
  - Reports raw MIB payload, rate-matching parameters, and CRC/bit-error counts inspired by MATLAB/Simulink LTE diagnostics; all decoding logic is our own Python implementation
- TDD extras (heuristic, presentation‑friendly):
  - Special subframe (subframe 1) detector via center-band energy
  - UL‑DL configuration index guesser (0..6) using first 5 subframes
- Notebook with 10+ figures: PSS/SSS correlations, PBCH spectra/scatter, energy heatmaps, LLR histograms, etc. (`notebooks/LTE_Analiz.ipynb`)

## Data
- You can supply IQ as `.raw`/`.iq` (interleaved float32) or `.mat` (standard or v7.3). Place it at the repo root or give a path on the CLI/notebook.
- For `.mat`, the loader auto-picks the first IQ-like array; override with `--key` on the CLI. If the file contains a scalar `fs`/`Fs`/`samp_rate`/`sample_rate`, it is used as a sampling-rate hint during auto-configuration.
- Conversion helper (optional): to generate a `.raw` from `.mat`, run:
  ```
  python scripts/mat_to_raw.py input.mat -o LTEIQ.raw --key IQ_variable_name
  ```
- Do not commit large/binary captures to Git — `.gitignore` excludes `*.raw`, `*.iq`, `*.bin`, `*.mat`, and `LTEIQ.raw`.

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
- Bandwidth auto-detection: the analyzer tries standard LTE bandwidths (1.4/3/5/10/15/20 MHz, normal CP) and picks the one with the strongest PSS match; you can still force behaviour via the CLI flags below.
- CLI quick run:
```
source yHHS_env/bin/activate
python scripts/run_analysis.py LTEIQ.raw           # or .iq
python scripts/run_analysis.py input.mat --key IQ  # MATLAB input with optional variable name
python scripts/run_analysis.py LTEIQ.raw --no-bruteforce          # skip PBCH search for speed
python scripts/run_analysis.py LTEIQ.raw --bruteforce-limit 40    # cap candidate NCellID search
# `.mat` files with `fs`/`Fs`/`samp_rate`/`sample_rate` scalars provide a sampling-rate hint for auto bandwidth selection.
```

- Notebook:
```
source yHHS_env/bin/activate
jupyter notebook notebooks/LTE_Analiz.ipynb
```
Select kernel: `Python (yHHS_env)`.

### Example Output (single 5 ms file)
```
NDLRB: 50
NDLRB_from_MIB: 75
DuplexMode: FDD
CyclicPrefix: Normal
NCellID: 455
NID1: 151
NID2: 2
NSubframe: 0
FrameOffsetSamples: 11665
Estimated_CFO_rad_per_sample: -2.26e-05
PSS_metric: 0.967
SSS_metric: 0.994
MIB_BitErrors: 0
CellRefP: 2
PHICHDuration: Extended
Ng: 2
NFrame: 509
```
`analyze_lte_iq` now surfaces the decoded MIB payload alongside the bit-error count so you can judge reliability. In this capture the PBCH CRC passes, but the decoded PBCH PCI (167) differs from the synchronisation PCI (455), so additional validation is recommended when mismatches appear.

- With 5 ms input, the pipeline reliably outputs: `NDLRB`, `CyclicPrefix`, `DuplexMode`, `NCellID`, `NSubframe` (SSS now strictly 3GPP-compliant).
- The analyzer also reports `FrameOffsetSamples`, the sample index shift required to align the capture so that subframe 0 starts at sample 0.
- PBCH fields (`CellRefP`, `PHICHDuration`, `Ng`, `NFrame`) now try both a lightweight phase-only EQ and a CRS-based candidate; success still depends on SNR/channel selectivity. MIB diagnostics expose bit-error counts so you can gauge confidence before trusting the numbers.
- Heuristic parts (TDD special subframe/config index) are intended for presentation/triage; longer captures improve accuracy.
- Production-grade PBCH decoding additionally benefits from multi-frame combining and robust channel estimation.
- Processing steps are informed by MATLAB/Simulink LTE workflows, but every routine here is an independent Python implementation.

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
