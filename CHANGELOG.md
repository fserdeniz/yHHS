# Changelog

## [Unreleased]
### Added
- Formal changelog documenting repository modifications.

## [2025-10-16] - Standards Alignment Sprint
### Added
- Implemented 3GPP TS 36.211 §6.11.2 compliant SSS generator (deterministic x_s/x_c/x_z recursions, q/q′ → m₀/m₁).
- Added spec-aligned PBCH descrambler using LTE Gold sequence (TS 36.211 §6.6.1).
- Introduced PBCH sub-block interleaver, circular-buffer rate-matching, and tail-biting Viterbi decoder matching TS 36.212 §5.1.4.2 / §5.3.1.
- Added docs/code.md describing system architecture; paper.md consolidating scientific-style documentation.

### Changed
- Updated README to highlight spec-compliant SSS and note pending CRS-based equalisation for PBCH.
- Adjusted brute-force PBCH decoder to exhaust RVs and sliding windows with spec descrambling.
- Refinements to CRS masking to assume presence of antenna ports 0-3 per TS 36.211 §6.10.1.
- Notebook LTE_Analiz.ipynb expanded with additional plots and TDD heuristics.

### Known Issues
- PBCH CRC still fails on sample capture pending CRS-based channel estimation; MIB fields remain `None` in CLI output.
