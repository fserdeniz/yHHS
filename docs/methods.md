# Methods

Signal Model and Assumptions
- Downlink LTE, 10 MHz BW, sampling rate 15.36 MHz, FFT size 1024, Normal CP (7 symbols/slot).
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
