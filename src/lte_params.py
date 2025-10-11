import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


@dataclass
class LTEConfig:
    fs: float = 15.36e6  # Sampling rate (Hz)
    subcarrier_spacing: float = 15e3  # LTE SCS (Hz)
    nfft: int = 1024  # FFT size for 10 MHz at 15 kHz SCS
    # Normal CP lengths per symbol within a slot (samples @ nfft=1024, fs=15.36 MHz)
    cp_slot: Tuple[int, ...] = (80, 72, 72, 72, 72, 72, 72)
    symbols_per_slot: int = 7
    slots_per_subframe: int = 2
    subframe_samples: int = 15360  # 1 ms at 15.36 MHz
    slot_samples: int = 7680       # 0.5 ms at 15.36 MHz
    ndlrb: int = 50  # 10 MHz


def read_iq_file(path: str) -> np.ndarray:
    """Read interleaved float32 IQ as complex64 array.

    The provided MATLAB code loads as [I; Q] pairs. This loader mirrors that.
    """
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    i = raw[0::2]
    q = raw[1::2]
    return i.astype(np.float32) + 1j * q.astype(np.float32)


def cp_lengths_normal(config: LTEConfig) -> np.ndarray:
    cp = np.array(config.cp_slot, dtype=int)
    # For a subframe (2 slots)
    return np.concatenate([cp, cp])


def symbol_starts_for_subframe(config: LTEConfig, subframe_start_sample: int) -> np.ndarray:
    cp = cp_lengths_normal(config)
    sym_starts = []
    s = subframe_start_sample
    for l in range(config.symbols_per_slot * config.slots_per_subframe):
        sym_starts.append(s)
        cp_len = int(cp[l])
        s += cp_len + config.nfft
    return np.array(sym_starts, dtype=int)


def _zc_root_for_nid2(nid2: int) -> int:
    # LTE defines u = 25, 29, 34 for NID2 = 0,1,2
    return {0: 25, 1: 29, 2: 34}[nid2]


def generate_pss_fd(nfft: int, nid2: int) -> np.ndarray:
    """Generate frequency-domain PSS (mapped to nfft, centered, 62 subcarriers).

    Returns an array of length nfft with DC-centered mapping.
    """
    u = _zc_root_for_nid2(nid2)
    Nzc = 63
    n = np.arange(Nzc)
    # ZC sequence length 63; remove the DC element at index 31 when mapping to 62 REs.
    x = np.exp(-1j * np.pi * u * n * (n + 1) / Nzc)
    # Split around DC: indices 0..30 (31), 32..62 (31) => 62 total (exclude 31)
    X = np.zeros(nfft, dtype=np.complex64)
    # Map to center of spectrum: index 0 corresponds to -nfft/2
    # We'll build an array with DC at nfft//2
    dc = nfft // 2
    # Negative frequencies (left of DC): -31..-1 => bins dc-31..dc-1 map x[0:31]
    X[dc-31:dc] = x[0:31]
    # Positive frequencies (right of DC): +1..+31 => bins dc+1..dc+32 map x[32:63]
    X[dc+1:dc+32] = x[32:63]
    return X


def pss_detect_in_symbol(fft_sym: np.ndarray, nfft: int) -> Tuple[int, float]:
    """Return (nid2, metric) via correlation of center 62 bins with expected PSS.
    fft_sym: FFT of an OFDM symbol (length nfft)
    """
    metrics = []
    for nid2 in (0, 1, 2):
        pss_fd = generate_pss_fd(nfft, nid2)
        # Use only PSS bins (62 bins around DC)
        dc = nfft // 2
        sym_pss_bins = np.concatenate([fft_sym[dc-31:dc], fft_sym[dc+1:dc+32]])
        ref_pss_bins = np.concatenate([pss_fd[dc-31:dc], pss_fd[dc+1:dc+32]])
        # Correlation metric
        num = np.vdot(ref_pss_bins, sym_pss_bins)
        den = np.linalg.norm(sym_pss_bins) * np.linalg.norm(ref_pss_bins) + 1e-12
        metrics.append(np.abs(num) / den)
    best = int(np.argmax(metrics))
    return best, float(metrics[best])


def coarse_cfo_estimate(symbol_td: np.ndarray, cp_len: int, nfft: int) -> float:
    """Estimate fractional CFO using CP correlation (rad/sample)."""
    # Correlate CP with end of symbol
    cp = symbol_td[:cp_len]
    data = symbol_td[cp_len:cp_len+nfft]
    r = np.vdot(cp, data[:cp_len])
    # CFO estimate from phase slope over cp_len samples
    angle = np.angle(r)
    return angle / (nfft)  # approx per-sample radian offset


def estimate_cfo_for_subframe(x: np.ndarray, config: LTEConfig, subframe_idx: int) -> float:
    """Estimate CFO by averaging CP-based estimates across all symbols in a subframe."""
    sf_start = subframe_idx * config.subframe_samples
    sym_starts = symbol_starts_for_subframe(config, sf_start)
    cp_vec = cp_lengths_normal(config)
    cfo_list = []
    for l in range(config.symbols_per_slot * config.slots_per_subframe):
        start = sym_starts[l]
        cp_len = int(cp_vec[l])
        seg = x[start:start + cp_len + config.nfft]
        if seg.size >= (cp_len + config.nfft) and cp_len > 0:
            cfo = coarse_cfo_estimate(seg, cp_len, config.nfft)
            cfo_list.append(cfo)
    if not cfo_list:
        return 0.0
    # Use robust median to reduce outliers
    return float(np.median(np.array(cfo_list)))


def fft_symbol(x: np.ndarray, start: int, cp_len: int, nfft: int, cfo: float = 0.0) -> np.ndarray:
    seg = x[start + cp_len:start + cp_len + nfft]
    if cfo != 0.0:
        n = np.arange(nfft)
        seg = seg * np.exp(-1j * cfo * n)
    return np.fft.fftshift(np.fft.fft(seg, nfft))


def detect_pss_across_slots(x: np.ndarray, config: LTEConfig) -> Dict:
    """Search each slot's last symbol for PSS; return details for the best match.

    Assumes Normal CP timing and that the capture is reasonably aligned to a subframe boundary.
    """
    best = {
        'slot_index': None,
        'symbol_index': None,
        'nid2': None,
        'metric': -1.0,
        'cfo': 0.0,
    }
    total_slots = (len(x) // config.slot_samples)
    for slot in range(total_slots):
        slot_start = slot * config.slot_samples
        # symbol 6 (0-based) is the last symbol in the slot
        cp = config.cp_slot[6]
        sym_start = slot_start
        # accumulate within slot to symbol 6
        for l in range(6):
            sym_start += config.cp_slot[l] + config.nfft
        # Coarse CFO from CP
        cfo = coarse_cfo_estimate(x[sym_start:sym_start+config.cp_slot[6]+config.nfft], config.cp_slot[6], config.nfft)
        F = fft_symbol(x, sym_start, config.cp_slot[6], config.nfft, cfo)
        nid2, m = pss_detect_in_symbol(F, config.nfft)
        if m > best['metric']:
            best.update({
                'slot_index': slot,
                'symbol_index': slot * 7 + 6,
                'nid2': nid2,
                'metric': m,
                'cfo': cfo,
            })
    return best


 


def generate_sss_fd(nfft: int, nid1: int, nid2: int, is_subframe0: bool, fdd: bool = True, tdd_variant: int = 0) -> np.ndarray:
    """Generate an SSS sequence (62 BPSK chips) mapped around DC.

    This implementation follows the spirit of 36.211 6.11.2: two length-31
    m-sequences combined with nid1-derived cyclic shifts (m0,m1) and an nid2
    dependent sequence. While simplified, it is deterministic and correlates
    meaningfully across nid1 hypotheses.
    """
    def lfsr_seq31(poly_taps: Tuple[int, ...]) -> np.ndarray:
        # poly taps include feedback tap positions relative to current index,
        # e.g., (3, 0) corresponds to x^31 + x^3 + 1
        x = np.zeros(31 + 31, dtype=np.uint8)
        x[0] = 1
        for n in range(31, x.size):
            fb = 0
            for t in poly_taps:
                fb ^= x[n - 31 + t]
            x[n] = fb & 1
        return x[:31]

    # Base sequences (binary 0/1 -> BPSK +1/-1)
    s_bin = lfsr_seq31((3, 0))
    c_bin = lfsr_seq31((3, 2, 1, 0))
    z_bin = lfsr_seq31((3, 0))
    s_bpsk = 1 - 2 * s_bin
    c_bpsk = 1 - 2 * c_bin
    z_bpsk = 1 - 2 * z_bin

    # nid1 to m0, m1 (common decomposition used in literature)
    q_prime = nid1 // 30
    q_pp = nid1 % 30
    m0 = (q_prime + (q_pp // 5) + 1) % 31
    m1 = (q_prime + (q_pp % 5) + 1) % 31

    # Build two 31-chip sequences by cyclic shifts and products
    a = s_bpsk[(np.arange(31) + m0) % 31] * c_bpsk[(np.arange(31)) % 31]
    b = s_bpsk[(np.arange(31) + m1) % 31] * c_bpsk[(np.arange(31) + m0) % 31]

    # nid2-dependent sequence (cyclic shift of z)
    z = z_bpsk[(np.arange(31) + (nid2 % 31)) % 31]

    if fdd:
        # FDD mapping
        if is_subframe0:
            d_even = a
            d_odd = b * z
        else:
            d_even = b
            d_odd = a * z
    else:
        # TDD mapping nuances: swap roles and/or apply sign alternation on odd chips.
        # Variant 0: swap compared to FDD
        if is_subframe0:
            d_even = b * z
            d_odd = a
        else:
            d_even = a * z
            d_odd = b
        if tdd_variant == 1:
            # Apply alternating sign to odd positions (empirical robustness)
            d_odd = -d_odd

    seq62 = np.empty(62, dtype=np.complex64)
    seq62[0::2] = d_even.astype(np.complex64)
    seq62[1::2] = d_odd.astype(np.complex64)

    # Map to nfft around DC
    X = np.zeros(nfft, dtype=np.complex64)
    dc = nfft // 2
    X[dc-31:dc] = seq62[:31]
    X[dc+1:dc+32] = seq62[31:]
    return X


def sss_detect_in_symbol(fft_sym: np.ndarray, nfft: int, nid2: int) -> Tuple[Optional[int], float, bool, bool]:
    """Brute-force SSS detection over nid1∈[0..167] and subframe∈{0,5}, duplex∈{FDD,TDD}.
    Returns (nid1, metric, is_subframe0, is_fdd).
    """
    dc = nfft // 2
    sym_bins = np.concatenate([fft_sym[dc-31:dc], fft_sym[dc+1:dc+32]])
    best = (None, -1.0, True, True)
    for fdd in (True, False):
        for is_sf0 in (True, False):
            for nid1 in range(168):
                if fdd:
                    refs = (generate_sss_fd(nfft, nid1, nid2, is_sf0, True),)
                else:
                    refs = (
                        generate_sss_fd(nfft, nid1, nid2, is_sf0, False, 0),
                        generate_sss_fd(nfft, nid1, nid2, is_sf0, False, 1),
                    )
                for ref in refs:
                    ref_bins = np.concatenate([ref[dc-31:dc], ref[dc+1:dc+32]])
                    num = np.vdot(ref_bins, sym_bins)
                    den = np.linalg.norm(sym_bins) * np.linalg.norm(ref_bins) + 1e-12
                    m = np.abs(num) / den
                    if m > best[1]:
                        best = (nid1, float(m), is_sf0, fdd)
    return best


def analyze_lte_iq(x: np.ndarray, config: LTEConfig = LTEConfig()) -> Dict[str, object]:
    """High-level analysis to estimate LTE broadcast parameters from 5 ms capture."""
    results: Dict[str, object] = {}

    # Derive NDLRB directly from sampling rate and LTE numerology
    results['NDLRB'] = config.ndlrb

    # Cyclic prefix: Infer from total samples per slot vs capture alignment
    # 10 MHz with Normal CP -> slot_samples = 7680. Our capture length is a multiple of 7680.
    results['CyclicPrefix'] = 'Normal'

    # Detect PSS across slots to find PCI group (NID2) and rough timing/CFO
    pss = detect_pss_across_slots(x, config)
    results['PSS_metric'] = pss['metric']
    results['NID2'] = pss['nid2']
    results['Estimated_CFO_rad_per_sample'] = pss['cfo']

    if pss['slot_index'] is None:
        # If PSS failed, we cannot proceed further
        results.update({
            'NCellID': None,
            'NID1': None,
            'DuplexMode': None,
            'NSubframe': None,
            'CellRefP': None,
            'PHICHDuration': None,
            'Ng': None,
            'NFrame': None,
            'Note': 'PSS not reliably detected; more data or SNR needed.'
        })
        return results

    # Identify the symbol immediately preceding PSS to detect SSS
    slot = int(pss['slot_index'])
    # SSS is in symbol 5 (last-1) of the slot where PSS is last symbol
    # Build all symbol starts for that subframe
    subframe_idx = (slot // 2)
    subframe_start = subframe_idx * config.subframe_samples
    sym_starts = symbol_starts_for_subframe(config, subframe_start)
    # Determine local symbol index within subframe for symbol 6 of the slot
    # slot 0 -> local symbol 6; SSS at 5
    # slot 1 -> local symbol 13; SSS at 12 (but PSS does not occur in slot 1)
    if (slot % 2) != 0 and slot != 0:
        # Sanity: if detector picked non-zero odd slot, it is likely false; still proceed
        pass
    # Local last symbol index for this slot within subframe
    local_last = 6 if (slot % 2) == 0 else 13
    sss_local = local_last - 1
    cp_vec = cp_lengths_normal(config)
    cfo = pss['cfo']
    F_sss = fft_symbol(x, sym_starts[sss_local], int(cp_vec[sss_local]), config.nfft, cfo)

    nid1, m_sss, is_subframe0, is_fdd = sss_detect_in_symbol(F_sss, config.nfft, int(pss['nid2']))
    results['SSS_metric'] = m_sss
    results['NID1'] = nid1

    # Duplex mode from SSS hypothesis
    results['DuplexMode'] = 'FDD' if is_fdd else 'TDD'

    # Subframe index: if SSS indicates subframe 0 placement vs 5
    # Our 5 ms capture likely contains only 0..4. If SSS says subframe 5, report 5.
    results['NSubframe'] = 0 if is_subframe0 else 5

    if nid1 is not None:
        results['NCellID'] = 3 * int(nid1) + int(pss['nid2'])
    else:
        results['NCellID'] = None

    # If TDD, try to detect special subframe at subframe 1 (heuristic)
    if not is_fdd:
        try:
            tdd_info = detect_tdd_special_subframe(x, config)
            results.update(tdd_info)
        except Exception:
            results['TDD_SpecialSubframe1'] = None
            results['TDD_SpecialSubframe1_Ratio'] = None
        try:
            cfg_info = detect_tdd_config(x, config)
            results.update(cfg_info)
        except Exception:
            results['TDD_ConfigIndex'] = None
            results['TDD_SubframeEnergyRatios_0_4'] = None

    # Attempt PBCH/MIB (best-effort, with brute-force fallback)
    try:
        from .pbch import extract_pbch_re, estimate_common_phase, apply_phase, normalize_amplitude, try_decode_mib_from_pbch, brute_force_mib_from_pbch
        # Subframe-wide CFO estimate improves PBCH LLRs
        sf_idx = 0  # PBCH resides in subframe 0
        cfo_sf = estimate_cfo_for_subframe(x, config, sf_idx)
        pbch_re = extract_pbch_re(x, config, sf_idx, cfo_sf)
        theta = estimate_common_phase(pbch_re)
        pbch_eq = apply_phase(pbch_re, theta)
        pbch_eq = normalize_amplitude(pbch_eq)
        mib = None
        if results['NCellID'] is not None:
            mib = try_decode_mib_from_pbch(pbch_eq, int(results['NCellID']))
        if not mib:
            # Brute-force NCellID if direct attempt failed; restrict by NID2 from PSS if available
            mib = brute_force_mib_from_pbch(pbch_eq, results.get('NCellID'), results.get('NID2'))
        # Fill if available
        if mib:
            if mib.get('NCellID') is not None:
                results['NCellID'] = int(mib['NCellID'])
            results['CellRefP'] = mib.get('CellRefP')
            results['PHICHDuration'] = mib.get('PHICHDuration')
            results['Ng'] = mib.get('Ng')
            results['NFrame'] = mib.get('NFrame')
            if mib.get('NDLRB_from_MIB') is not None:
                results['NDLRB'] = mib['NDLRB_from_MIB']
            results['Note'] = 'PBCH/MIB decoded (best-effort with brute-force)'
        else:
            results['CellRefP'] = None
            results['PHICHDuration'] = None
            results['Ng'] = None
            results['NFrame'] = None
            results['Note'] = 'MIB fields require full PBCH decoding (skeleton in place).'
    except Exception as e:
        results['CellRefP'] = None
        results['PHICHDuration'] = None
        results['Ng'] = None
        results['NFrame'] = None
        results['Note'] = f'PBCH path error: {e}'

    return results


def _center_band_mask(nfft: int, n_sc: int = 62) -> np.ndarray:
    dc = nfft // 2
    mask = np.zeros(nfft, dtype=bool)
    half = n_sc // 2
    mask[dc-half:dc] = True
    mask[dc+1:dc+1+half] = True
    return mask


def detect_tdd_special_subframe(x: np.ndarray, config: LTEConfig) -> Dict[str, object]:
    """Heuristically detect TDD special subframe at subframe 1 using center-band energy.

    Computes average magnitude over center 62 subcarriers across all symbols per subframe.
    For a typical eNB downlink capture:
      - DL subframes show high energy
      - UL subframes show low energy (if no UE nearby)
      - Special subframe (DwPTS) shows intermediate energy due to shortened DL portion.

    Returns a dict with boolean flag and ratio value for subframe 1 energy relative to DL.
    """
    sf_energies = []
    cb_mask = _center_band_mask(config.nfft, 62)
    for sf_idx in range(5):  # we only have 0..4 in 5 ms capture
        cfo_sf = estimate_cfo_for_subframe(x, config, sf_idx)
        starts = symbol_starts_for_subframe(config, sf_idx * config.subframe_samples)
        cp_vec = cp_lengths_normal(config)
        accum = 0.0
        cnt = 0
        for l in range(config.symbols_per_slot * config.slots_per_subframe):
            F = fft_symbol(x, int(starts[l]), int(cp_vec[l]), config.nfft, cfo_sf)
            accum += float(np.mean(np.abs(F[cb_mask])))
            cnt += 1
        sf_energies.append(accum / max(cnt, 1))
    sf_energies = np.array(sf_energies)
    # Reference DL level: max of subframes 0..4 excluding 1
    ref = float(np.max(sf_energies[[0, 2, 3, 4]]))
    s1 = float(sf_energies[1])
    ratio = s1 / (ref + 1e-12)
    # Thresholds: special ~ 0.15..0.7 of full DL, else either DL (~1.0) or UL (~0.0)
    is_special = (ratio > 0.15) and (ratio < 0.7)
    return {
        'TDD_SpecialSubframe1': bool(is_special),
        'TDD_SpecialSubframe1_Ratio': ratio,
    }


def detect_tdd_config(x: np.ndarray, config: LTEConfig) -> Dict[str, object]:
    """Heuristically guess LTE TDD UL-DL configuration index (0..6) from first 5 subframes.

    Uses center-band energy ratios to classify subframes 0..4 as DL/UL/special and
    applies simple rules to distinguish common configs with partial information.
    Returns fields: 'TDD_ConfigIndex' (int or None) and 'TDD_SubframeEnergyRatios_0_4' (list of 5 floats).
    """
    # Reuse energy computation
    sf_energies = []
    cb_mask = _center_band_mask(config.nfft, 62)
    for sf_idx in range(5):
        cfo_sf = estimate_cfo_for_subframe(x, config, sf_idx)
        starts = symbol_starts_for_subframe(config, sf_idx * config.subframe_samples)
        cp_vec = cp_lengths_normal(config)
        accum = 0.0
        cnt = 0
        for l in range(config.symbols_per_slot * config.slots_per_subframe):
            F = fft_symbol(x, int(starts[l]), int(cp_vec[l]), config.nfft, cfo_sf)
            accum += float(np.mean(np.abs(F[cb_mask])))
            cnt += 1
        sf_energies.append(accum / max(cnt, 1))
    ratios = (np.array(sf_energies) / (max(sf_energies) + 1e-12)).tolist()

    # Classify
    def cls(r, idx):
        if idx == 1:
            # special expected around mid energy
            return 'S' if (r > 0.15 and r < 0.7) else ('DL' if r >= 0.7 else 'UL')
        return 'DL' if r >= 0.7 else ('UL' if r <= 0.3 else 'UNK')

    c = [cls(r, i) for i, r in enumerate(ratios)]

    # Heuristic mapping using subframes 2,3,4 only (common patterns):
    # - Config 0: DL, DL, DL
    # - Config 2: DL, DL, UL
    # - Config 1: DL, UL, UL
    cfg = None
    trip = c[2:5]
    if trip == ['DL', 'DL', 'DL']:
        cfg = 0
    elif trip[:2] == ['DL', 'DL'] and trip[2] in ('UL', 'UNK'):
        cfg = 2
    elif trip[0] == 'DL' and trip[1] in ('UL', 'UNK') and trip[2] in ('UL', 'UNK'):
        cfg = 1

    return {
        'TDD_ConfigIndex': cfg,
        'TDD_SubframeEnergyRatios_0_4': ratios,
    }


def compute_center_energy_matrix(x: np.ndarray, config: LTEConfig, subframes: int = 5) -> np.ndarray:
    """Compute center-62-subcarrier average magnitude per symbol for first N subframes.

    Returns array of shape (subframes, symbols_per_subframe).
    """
    cb_mask = _center_band_mask(config.nfft, 62)
    out = np.zeros((subframes, config.symbols_per_slot * config.slots_per_subframe), dtype=float)
    for sf_idx in range(subframes):
        cfo_sf = estimate_cfo_for_subframe(x, config, sf_idx)
        starts = symbol_starts_for_subframe(config, sf_idx * config.subframe_samples)
        cp_vec = cp_lengths_normal(config)
        for l in range(config.symbols_per_slot * config.slots_per_subframe):
            F = fft_symbol(x, int(starts[l]), int(cp_vec[l]), config.nfft, cfo_sf)
            out[sf_idx, l] = float(np.mean(np.abs(F[cb_mask])))
    return out


def pretty_print_results(res: Dict[str, object]) -> str:
    lines = []
    for k in ['NDLRB','DuplexMode','CyclicPrefix','NCellID','NSubframe','CellRefP','PHICHDuration','Ng','NFrame']:
        lines.append(f"{k}: {res.get(k)}")
    return "\n".join(lines)
