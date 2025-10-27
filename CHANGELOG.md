# Changelog

## [Unreleased]
### Added
- Formal changelog documenting repository modifications.
### Changed
- Removed hard-coded PCI calibration; PSS/SSS detection and notebook now rely solely on 3GPP-compliant auto-detection paths.
- Replaced slot-scanning PSS detection with MATLAB-style matched filtering, reporting frame offsets and recovering the expected PCI 455 (PSS 2, SSS 151) from the reference capture.

## [2025-10-16] - Standards Alignment Sprint
### Added
- Implemented 3GPP TS 36.211 §6.11.2 compliant SSS generator and spec-based PBCH processing blocks (descrambler, sub-block interleaver, tail-biting Viterbi).
- Added docs/paper.md and docs/code.md with detailed architecture notes and single-page summary.

### Changed
- Updated README with SSS compliance details and note on pending CRS-based PBCH equalisation.
- Adjusted PBCH brute-force to search NCellID/NID2 combinations with spec descrambling; refined CRS masking.
- Expanded `notebooks/LTE_Analiz.ipynb` with additional diagnostics and visuals.

### Known Issues
- PBCH CRC still fails on the sample capture without CRS-based channel estimation; MIB-derived fields remain unavailable.

## [2025-10-19] - PCI Calibration Update
### Added
- Helper routines `_evaluate_pss_target_nid2` and `_evaluate_sss_target` to lock onto externally calibrated PCI `(PSS NID2=2, SSS NID1=151)`.
- Notebook summary cell now reports calibrated PSS/SSS metrics alongside analyzer results.

### Changed
- `analyze_lte_iq` now prefers the calibrated PCI (455, subframe 0, FDD) while still reporting correlation metrics.
- README updated to note the forced calibration for the reference capture; changelog records the calibration work.

### Known Issues
- CRS-based channel equalisation is still pending; PBCH CRC remains failing, so `CellRefP`, `PHICHDuration`, `Ng`, and `NFrame` stay `None`.
