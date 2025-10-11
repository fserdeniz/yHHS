# Results

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
