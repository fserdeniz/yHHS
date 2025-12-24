# LTE Broadcast Parameter Extraction from a 5 ms LTE IQ Capture

## Abstract

We present a Python-based pipeline to extract LTE broadcast parameters from a short, 5 ms IQ capture (Fs=15.36 MHz, 10 MHz BW). The system detects PSS/SSS to estimate cell synchronization and identity, and performs a best-effort PBCH/MIB decoding to recover system broadcast parameters. Key contributions include a deterministic SSS generator with FDD/TDD variants, robust PBCH resource extraction with improved CRS masking (ports 0/1), a spec PBCH scrambler (`c_init=(NCellID<<9)|(i_mod4<<5)|0x1FF`) with guarded variants, constrained brute-force over NCellID guided by PSS (NID2), and simple equalization via common phase estimation plus optional CRS-based PBCH equalization. With a single 5 ms capture, the pipeline typically recovers NDLRB, DuplexMode, CyclicPrefix, NCellID, NSubframe, and often MIB fields (CellRefP, PHICHDuration, Ng, NFrame), subject to SNR and alignment. The implementation emphasizes clarity and reproducibility suitable for educational demonstration and lab analysis.

## Introduction

LTE broadcast parameters (e.g., NDLRB, DuplexMode, CyclicPrefix, NCellID, MIB fields) are conveyed by synchronization and broadcast channels: PSS, SSS, and PBCH. Recovering these from short IQ captures helps with spectrum exploration, lab exercises, and reverse engineering. This project targets a 5 ms capture at 15.36 MHz for a 10 MHz LTE downlink and implements a practical, didactic pipeline.

Goals
- Robustly detect PSS/SSS and infer `NCellID = 3*NID1 + NID2`.
- Estimate `DuplexMode` and the likely `NSubframe` (0 or 5) from SSS.
- Attempt PBCH demodulation to decode MIB and recover `CellRefP`, `PHICHDuration`, `Ng`, and `NFrame`.

Related Work
- 3GPP specifications define the exact signals and procedures [1][2][11].
- Accessible tutorials (ShareTechnote) provide intuition and reference formulas for PSS/SSS/PBCH/MIB [1–8].

Scope
- Single 5 ms snapshot, no multi-frame combining.
- Emphasis on clarity and reproducibility over full spec coverage.

## Methods

Signal Model and Assumptions
- Downlink LTE, standard bandwidth set {1.4, 3, 5, 10, 15, 20} MHz (NDLRB ∈ {6,15,25,50,75,100}); the analyzer auto-selects the best-fitting bandwidth via PSS correlation. Sampling rates/NFFT follow LTE numerology (e.g., 30.72 Msps/2048 for 20 MHz). Normal CP (7 symbols/slot).
- Single 5 ms capture (5 subframes), unknown CFO and phase.

Acquisition and Synchronization
- PSS search across slots: compute FFT of last symbol per slot and correlate 62 central subcarriers with generated PSS for NID2∈{0,1,2} [1].
- SSS search on the symbol preceding PSS: deterministic SSS generator with FDD and TDD variants, brute-force over NID1∈[0..167], subframe∈{0,5} to maximize normalized correlation [2].

Frequency/Phase Correction
- Coarse CFO via CP correlation per symbol; median across a subframe for robustness.
- Common phase estimation on PBCH REs via unit-vector averaging; apply phase rotation.

PBCH Resource Extraction and LLRs
- Extract PBCH REs from slot 1, symbols 0..3 over center 6 RB (72 subcarriers) [3].
- Mask CRS: port-0 pattern (k+v_shift) mod 6 = 0 at l=0; optionally port-1 shift (+3) [6].
- Map QPSK symbols to soft LLRs per bit (Re/Im) under unit-energy assumption.

De‑rate Matching and Decoding
- Standard-inspired 480→120 softbit de‑rate matching per redundancy version i_mod4∈{0..3}.
- Descrambling: several Gold-sequence c_init variants tied to `NCellID` and `i_mod4`.
- Viterbi (rate-1/3, K=7) on 120 softbits → 40 bits; split into 24-bit payload + 16-bit CRC.
- CRC checks: several cell-dependent masks; accept candidates that also yield plausible MIB fields.

Brute‑Force Strategy
- If direct `NCellID` fails: iterate `NCellID` over PCI set filtered by measured NID2 to reduce search.
- Reduce runtime with quality gates: limited scrambler variants, 2–3 windows, early rejection on low |LLR| mean.

TDD Heuristics
- Special subframe detection (subframe 1) from center-band energy ratio relative to downlink subframes.
- UL‑DL configuration guess from subframes 2–4 energy classification (DL/UL/UNK).

Implementation
- Core: `src/lte_params.py`, `src/pbch.py`. CLI: `scripts/run_analysis.py`. Notebook: `notebooks/LTE_Analiz.ipynb`.

## Results

Dataset
- Single file `LTEIQ.raw` (provided with the assignment): 5 ms, 10 MHz BW, Fs=15.36 MHz.

Key Outcomes (example run)
- `NDLRB: 15`, `DuplexMode: FDD`, `CyclicPrefix: Normal`.
- `NCellID: 7`, `NSubframe: 0`.
- `CellRefP: 2`, `PHICHDuration: Extended`, `Ng: 1/2`, `NFrame: 109`.

Robustness
- PSS/SSS: Stable across moderate SNRs due to normalized correlation.
- PBCH/MIB: Best-effort; success depends on SNR, channel selectivity, and capture alignment. The constrained brute-force with quality gating mitigates false positives.

Runtime
- Typical runtime: ~10–60 s on a modern CPU for full analysis with brute-force enabled.

## Discussion

Strengths
- Works with a single 5 ms capture and no prior alignment.
- Deterministic SSS generator (FDD/TDD) improves NCellID identification.
- PBCH decoding includes practical steps (CRS masking, descrambler variants, quality gates) that increase success rate without overcomplexity.

Limitations
- Lightweight equalization (phase-only plus an optional CRS-based candidate) may still be insufficient for highly frequency-selective channels.
- CRC mask family is heuristic; full spec-accurate PBCH CRC treatment can further reduce false positives.
- Single-transmission view; combining across 4 PBCH repetitions (40 ms) would improve reliability.

Future Work
- Integrate CRS-based channel estimation and interpolation over PBCH bandwidth.
- Full 36.212 rate-matching/interleaver implementation and tighter PBCH CRC/scrambler handling beyond the current c_init coverage.
- Soft-combining across multiple captures; blind NCellID search with stronger priors.

## Conclusion

We demonstrated a practical LTE broadcast parameter extraction pipeline effective on a single 5 ms capture. The approach balances scientific rigor and pragmatic heuristics: deterministic synchronization, lightweight equalization, and constrained brute-force for PBCH/MIB. The resulting system produces reliable synchronization and cell identity along with frequent recovery of MIB fields. The codebase and notebook are intended for teaching, prototyping, and lab analysis, and provide a solid foundation for integrating more spec-accurate components.

## Reproducibility

Code and Environment
- Python 3.10+
- Dependencies in `requirements.txt`
- Recommended venv: `yHHS_env`

Steps
- Place IQ at repo root (`LTEIQ.raw`/`.iq` interleaved float32 or `.mat`; use `--key` to pick a MATLAB variable when needed). If your `.mat` contains a scalar `fs`/`Fs`/`samp_rate`/`sample_rate`, it will be used as a sampling-rate hint for auto-configuration.
- Create venv and install dependencies:
  - `python3 -m venv yHHS_env`
  - `source yHHS_env/bin/activate`
  - `pip install -r requirements.txt`
 - Run CLI: `python scripts/run_analysis.py LTEIQ.raw` (add `--no-bruteforce` for speed or `--bruteforce-limit N` to cap PBCH search; `.mat` supported with `--key VAR`)
- Or open the notebook: `notebooks/LTE_Analiz.ipynb`

Determinism
- The pipeline is deterministic given the same input file and environment.
- Randomness is not used in detection or decoding steps.

Data Availability
- `LTEIQ.raw` is not tracked in Git. Ensure the file path matches the code/notebook expectations.

## References

- [1] ShareTechnote, "LTE PSS – Primary Synchronization Signal." Online: https://www.sharetechnote.com/html/Handbook_LTE_PSS.html
- [2] ShareTechnote, "LTE SSS – Secondary Synchronization Signal." Online: https://www.sharetechnote.com/html/Handbook_LTE_SSS.html
- [3] ShareTechnote, "LTE PBCH – Physical Broadcast Channel." Online: https://www.sharetechnote.com/html/Handbook_LTE_PBCH.html
- [4] ShareTechnote, "LTE MIB – Master Information Block." Online: https://www.sharetechnote.com/html/Handbook_LTE_MIB.html
- [5] ShareTechnote, "LTE SIB – System Information Block." Online: https://www.sharetechnote.com/html/Handbook_LTE_SIB.html
- [6] ShareTechnote, "LTE Reference Signal (RS/CRS)." Online: https://www.sharetechnote.com/html/Handbook_LTE_ReferenceSignal.html
- [7] ShareTechnote, "LTE PCFICH – Physical Control Format Indicator Channel." Online: https://www.sharetechnote.com/html/Handbook_LTE_PCFICH.html
- [8] ShareTechnote, "LTE PDCCH – Physical Downlink Control Channel." Online: https://www.sharetechnote.com/html/Handbook_LTE_PDCCH.html
- [9] 3GPP TS 36.211, "E-UTRA; Physical channels and modulation." Online: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=2425
- [10] 3GPP TS 36.212, "E-UTRA; Multiplexing and channel coding." Online: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=2426
- [11] 3GPP TS 36.300, "E-UTRA and E-UTRAN; Overall description; Stage 2." Online: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=2430
