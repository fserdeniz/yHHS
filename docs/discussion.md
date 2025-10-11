# Discussion

Strengths
- Works with a single 5 ms capture and no prior alignment.
- Deterministic SSS generator (FDD/TDD) improves NCellID identification.
- PBCH decoding includes practical steps (CRS masking, descrambler variants, quality gates) that increase success rate without overcomplexity.

Limitations
- Simplified equalization (common phase + per-symbol normalization) may be insufficient for highly frequency-selective channels.
- CRC mask family is heuristic; full spec-accurate PBCH CRC treatment can further reduce false positives.
- Single-transmission view; combining across 4 PBCH repetitions (40 ms) would improve reliability.

Future Work
- Integrate CRS-based channel estimation and interpolation over PBCH bandwidth.
- Full 36.212 rate-matching/interleaver implementation and spec-exact PBCH scrambling.
- Soft-combining across multiple captures; blind NCellID search with stronger priors.
