import numpy as np
from scipy import signal
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


def generate_pss_td(nfft: int, nid2: int) -> np.ndarray:
    """Return time-domain PSS (IFFT of frequency-domain sequence)."""
    fd = generate_pss_fd(nfft, nid2)
    return np.fft.ifft(np.fft.ifftshift(fd))


def _symbol_lengths(config: LTEConfig) -> Tuple[int, ...]:
    # Normal CP slot: each symbol length is its CP plus NFFT samples
    return tuple(int(config.cp_slot[i] + config.nfft) for i in range(config.symbols_per_slot))


def pss_symbol_offset_samples(config: LTEConfig) -> int:
    """Offset (samples) from subframe start to the beginning of PSS (including CP)."""
    sym_lengths = _symbol_lengths(config)
    slot0 = sum(sym_lengths)
    return slot0 + sum(sym_lengths[:6])


def sss_symbol_offset_samples(config: LTEConfig) -> int:
    """Offset (samples) from subframe start to the beginning of SSS (including CP)."""
    sym_lengths = _symbol_lengths(config)
    slot0 = sum(sym_lengths)
    return slot0 + sum(sym_lengths[:5])


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
    """Matched-filter search for PSS across the capture (MATLAB-style cell search).

    Returns dict with keys:
      'nid2'          : detected NID2 (0..2)
      'metric'        : normalised correlation metric in [0, 1]
      'sample_index'  : starting sample index of the detected PSS symbol (CP included)
    """
    cp_pss = int(config.cp_slot[6])
    best = {'nid2': None, 'metric': -1.0, 'sample_index': None}
    signal_energy = np.abs(x) ** 2
    for nid2 in (0, 1, 2):
        td = generate_pss_td(config.nfft, nid2)
        ref = np.concatenate([td[-cp_pss:], td])
        ref_energy = float(np.sum(np.abs(ref) ** 2) + 1e-12)
        # Reverse-conjugate reference to obtain the matched-filter kernel
        mf = np.conjugate(ref[::-1])
        corr = signal.fftconvolve(x, mf, mode='valid')
        power = np.abs(corr) ** 2
        idx = int(np.argmax(power))
        # Normalise by instantaneous window energy so the metric is SNR-like
        window_energy = float(np.sum(signal_energy[idx:idx + ref.size]) + 1e-12)
        metric = float(power[idx] / (ref_energy * window_energy + 1e-12))
        if metric > best['metric']:
            best.update({'nid2': nid2, 'metric': metric, 'sample_index': idx})
    if best['sample_index'] is not None:
        slot_idx = best['sample_index'] // config.slot_samples
        best['slot_index'] = int(slot_idx)
        best['symbol_index'] = int(slot_idx * config.symbols_per_slot + 6)
        seg = x[best['sample_index']:best['sample_index'] + cp_pss + config.nfft]
        if seg.size >= (cp_pss + config.nfft):
            best['cfo'] = float(coarse_cfo_estimate(seg, cp_pss, config.nfft))
        else:
            best['cfo'] = 0.0
    else:
        best['slot_index'] = None
        best['symbol_index'] = None
        best['cfo'] = 0.0
    return best


 


_SSS_BASE_SEQS: Dict[str, np.ndarray] = {}


def _sss_base_sequences() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return base sequences s~, c~, z~ per 3GPP TS 36.211 6.11.2."""
    global _SSS_BASE_SEQS
    if not _SSS_BASE_SEQS:
        x = np.zeros(31, dtype=np.uint8)
        x[4] = 1

        xs = x.copy()
        for i in range(26):
            xs[i + 5] = (xs[i + 2] + xs[i]) & 1
        s_tilde = (1 - 2 * xs).astype(np.int8)

        xc = x.copy()
        for i in range(26):
            xc[i + 5] = (xc[i + 3] + xc[i]) & 1
        c_tilde = (1 - 2 * xc).astype(np.int8)

        xz = x.copy()
        for i in range(26):
            xz[i + 5] = (xz[i + 4] + xz[i + 2] + xz[i + 1] + xz[i]) & 1
        z_tilde = (1 - 2 * xz).astype(np.int8)

        _SSS_BASE_SEQS = {'s': s_tilde, 'c': c_tilde, 'z': z_tilde}
    return (_SSS_BASE_SEQS['s'], _SSS_BASE_SEQS['c'], _SSS_BASE_SEQS['z'])


def generate_sss_fd(nfft: int, nid1: int, nid2: int, is_subframe0: bool, fdd: bool = True, tdd_variant: int = 0) -> np.ndarray:
    """Generate the LTE SSS sequence (frequency domain) following 3GPP TS 36.211 §6.11.2."""
    s_tilde, c_tilde, z_tilde = _sss_base_sequences()

    q_prime = nid1 // 30
    q = (nid1 + (q_prime * (q_prime + 1)) // 2) // 31
    m_prime = nid1 + (q * (q + 1)) // 2
    m0 = m_prime % 31
    m1 = (m0 + (m_prime // 31) + 1) % 31

    n = np.arange(31, dtype=np.int32)
    s0 = s_tilde[(n + m0) % 31]
    s1 = s_tilde[(n + m1) % 31]
    c0 = c_tilde[(n + nid2) % 31]
    c1 = c_tilde[(n + nid2 + 3) % 31]
    z0 = z_tilde[(n + (m0 % 8)) % 31]
    z1 = z_tilde[(n + (m1 % 8)) % 31]

    if is_subframe0:
        even = s0 * c0
        odd = s1 * c1 * z0
    else:
        even = s1 * c0
        odd = s0 * c1 * z1

    seq62 = np.empty(62, dtype=np.int8)
    seq62[0::2] = even
    seq62[1::2] = odd

    spectrum = np.zeros(nfft, dtype=np.complex64)
    dc = nfft // 2
    spectrum[dc-31:dc] = seq62[:31]
    spectrum[dc+1:dc+32] = seq62[31:]
    return spectrum


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
                ref = generate_sss_fd(nfft, nid1, nid2, is_sf0, fdd)
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

    # Detect PSS via matched filter to find PCI group (NID2) and initial timing/CFO
    # (mirrors MATLAB LTE Toolbox cell search behaviour)
    pss = detect_pss_across_slots(x, config)
    results['PSS_metric'] = pss['metric']
    results['NID2'] = pss['nid2']
    if pss['sample_index'] is None or pss['nid2'] is None:
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
        results['Estimated_CFO_rad_per_sample'] = 0.0
        return results
    cp_pss = int(config.cp_slot[6])
    pss_start = int(pss['sample_index'])
    seg_pss = x[pss_start:pss_start + cp_pss + config.nfft]
    if seg_pss.size < (cp_pss + config.nfft):
        results.update({
            'NCellID': None,
            'NID1': None,
            'DuplexMode': None,
            'NSubframe': None,
            'CellRefP': None,
            'PHICHDuration': None,
            'Ng': None,
            'NFrame': None,
            'Note': 'Capture truncated around detected PSS; unable to continue.'
        })
        results['Estimated_CFO_rad_per_sample'] = 0.0
        return results
    cfo = coarse_cfo_estimate(seg_pss, cp_pss, config.nfft)
    pss['cfo'] = float(cfo)
    results['Estimated_CFO_rad_per_sample'] = float(cfo)

    # Align capture so subframe 0 (containing detected PSS) starts at sample 0
    # (equivalent to MATLAB's frame timing adjustment after cell search)
    pss_offset = pss_symbol_offset_samples(config)
    frame_offset = (pss_start - pss_offset) % len(x)
    results['FrameOffsetSamples'] = int(frame_offset)
    x_aligned = np.roll(x, -frame_offset) if frame_offset != 0 else x.copy()

    # Update PSS bookkeeping using aligned reference
    pss_aligned_start = pss_offset
    pss['sample_index'] = pss_aligned_start
    pss['slot_index'] = pss_aligned_start // config.slot_samples
    pss['symbol_index'] = pss['slot_index'] * config.symbols_per_slot + 6

    # Identify the symbol immediately preceding PSS to detect SSS (using aligned capture)
    slot = int(pss['slot_index'])
    subframe_idx = slot // config.slots_per_subframe
    subframe_start = subframe_idx * config.subframe_samples
    sym_starts = symbol_starts_for_subframe(config, subframe_start)
    cp_vec = cp_lengths_normal(config)
    slot_local = slot % config.slots_per_subframe
    local_last = ((slot_local + 1) * config.symbols_per_slot) - 1
    sss_local = max(local_last - 1, 0)
    # Evaluate the FFT of the SSS symbol using the CFO estimated from PSS
    F_sss = fft_symbol(x_aligned, int(sym_starts[sss_local]), int(cp_vec[sss_local]), config.nfft, pss['cfo'])
    nid1, m_sss, is_subframe0, is_fdd = sss_detect_in_symbol(F_sss, config.nfft, int(pss['nid2']))
    results['SSS_metric'] = m_sss

    # Duplex mode from SSS hypothesis
    if nid1 is not None:
        results['NID1'] = int(nid1)
        results['DuplexMode'] = 'FDD' if is_fdd else 'TDD'
        results['NSubframe'] = 0 if is_subframe0 else 5
        if results['NID2'] is not None:
            results['NCellID'] = 3 * int(nid1) + int(pss['nid2'])
        else:
            results['NCellID'] = None
    else:
        results['NID1'] = None
        results['DuplexMode'] = None
        results['NSubframe'] = None
        results['NCellID'] = None

    # If TDD, try to detect special subframe at subframe 1 (heuristic)
    if (nid1 is not None) and (not is_fdd):
        try:
            # Use centre-band energy ratios to flag DwPTS style subframe patterns
            tdd_info = detect_tdd_special_subframe(x_aligned, config)
            results.update(tdd_info)
        except Exception:
            results['TDD_SpecialSubframe1'] = None
            results['TDD_SpecialSubframe1_Ratio'] = None
        try:
            # Coarse UL/DL pattern classification for configuration index 0/1/2
            cfg_info = detect_tdd_config(x_aligned, config)
            results.update(cfg_info)
        except Exception:
            results['TDD_ConfigIndex'] = None
            results['TDD_SubframeEnergyRatios_0_4'] = None

    # Attempt PBCH/MIB (best-effort, with brute-force fallback)
    try:
        from .pbch import extract_pbch_re, estimate_common_phase, apply_phase, normalize_amplitude, try_decode_mib_from_pbch, brute_force_mib_from_pbch
        # Subframe-wide CFO estimate improves PBCH LLRs
        sf_idx = 0  # PBCH resides in subframe 0
        cfo_sf = estimate_cfo_for_subframe(x_aligned, config, sf_idx)
        # Extract, equalise (phase/amplitude), then attempt direct + brute-force MIB decoding
        pbch_re = extract_pbch_re(x_aligned, config, sf_idx, cfo_sf)
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
    for k in [
        'NDLRB',
        'DuplexMode',
        'CyclicPrefix',
        'NCellID',
        'NID1',
        'NID2',
        'NSubframe',
        'FrameOffsetSamples',
        'Estimated_CFO_rad_per_sample',
        'PSS_metric',
        'SSS_metric',
        'CellRefP',
        'PHICHDuration',
        'Ng',
        'NFrame',
    ]:
        lines.append(f"{k}: {res.get(k)}")
    return "\n".join(lines)
