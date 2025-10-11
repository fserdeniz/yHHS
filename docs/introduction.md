# Introduction

LTE broadcast parameters (e.g., NDLRB, DuplexMode, CyclicPrefix, NCellID, MIB fields) are conveyed by synchronization and broadcast channels: PSS, SSS, and PBCH. Recovering these from short IQ captures helps with spectrum exploration, lab exercises, and reverse engineering. This project targets a 5 ms capture at 15.36 MHz for a 10 MHz LTE downlink and implements a practical, didactic pipeline.

Goals
- Robustly detect PSS/SSS and infer `NCellID = 3*NID1 + NID2`.
- Estimate `DuplexMode` and the likely `NSubframe` (0 or 5) from SSS.
- Attempt PBCH demodulation to decode MIB and recover `CellRefP`, `PHICHDuration`, `Ng`, and `NFrame`.

Related Work
- 3GPP specifications define the exact signals and procedures [1][2][4].
- Accessible tutorials (ShareTechnote) provide intuition and reference formulas for PSS/SSS/PBCH/MIB [1–3][4–8].

Scope
- Single 5 ms snapshot, no multi-frame combining.
- Emphasis on clarity and reproducibility over full spec coverage.
