import numpy as np
from typing import Optional, Tuple, Dict, List
from .lte_params import LTEConfig, symbol_starts_for_subframe, cp_lengths_normal, fft_symbol


def _center_band_indices(nfft: int, n_sc: int = 72) -> Tuple[np.ndarray, np.ndarray]:
    """Return negative and positive frequency indices around DC totaling n_sc.
    For even n_sc, excludes DC.
    """
    dc = nfft // 2
    half = n_sc // 2
    neg = np.arange(dc - half, dc, dtype=int)  # half bins
    pos = np.arange(dc + 1, dc + 1 + half, dtype=int)  # half bins
    return neg, pos


def extract_pbch_re(x: np.ndarray, cfg: LTEConfig, subframe_idx: int, cfo: float = 0.0) -> np.ndarray:
    """Extract PBCH resource elements from subframe 0, symbols 0..3 of slot 1 (Normal CP).

    Returns complex REs array of shape (4, 72) covering central 6 RB.
    NOTE: No CRS RE puncturing is applied in this simplified extractor.
    """
    if cfg.symbols_per_slot != 7:
        raise ValueError("Only Normal CP supported in this simplified PBCH extractor.")
    # PBCH exists only in subframe 0. If another subframe is passed, we still extract for demo.
    sf_start = subframe_idx * cfg.subframe_samples
    sym_starts = symbol_starts_for_subframe(cfg, sf_start)
    cp_vec = cp_lengths_normal(cfg)
    # Slot 1 local symbols: 7..13; PBCH on symbols 0..3 within slot 1 -> local 7..10
    idxs = [7, 8, 9, 10]
    re_list = []
    neg, pos = _center_band_indices(cfg.nfft, 72)
    for l in idxs:
        F = fft_symbol(x, int(sym_starts[l]), int(cp_vec[l]), cfg.nfft, cfo)
        band = np.concatenate([F[neg], F[pos]])
        re_list.append(band)
    return np.stack(re_list, axis=0)  # (4, 72)


def crs_data_mask_for_pbch(ncellid: int, cellrefp: int = 1) -> np.ndarray:
    """Return boolean mask of shape (4,72) for PBCH REs, masking CRS REs.

    Normal CP, PBCH occupies slot 1, symbols l=0..3. Cell-specific RS (CRS)
    for LTE appear on l=0 and l=4 within each slot for port 0; only l=0 overlaps PBCH.

    - For port 0: (k + v_shift) mod 6 == 0 at l = 0
    - For port 1 (when cellrefp >= 2): (k + v_shift + 3) mod 6 == 0 at l = 0
    - For ports 2/3 (when cellrefp == 4), CRS are on other symbols which do not
      overlap PBCH symbols 0..3 for Normal CP. We conservatively remove only l=0
      patterns for ports 0 and 1.
    """
    v_shift = ncellid % 6
    mask = np.ones((4, 72), dtype=bool)
    idxs = np.arange(72)
    crs0 = (idxs % 6) == (v_shift % 6)
    crs1 = (idxs % 6) == ((v_shift + 3) % 6)
    mask[0, crs0] = False  # port 0
    mask[0, crs1] = False  # port 1 (reserved even if unused)
    mask[1, crs0] = False  # port 2
    mask[1, crs1] = False  # port 3
    return mask


def estimate_common_phase(pbch_re: np.ndarray) -> float:
    """Estimate common phase using unit-vector averaging across all PBCH REs.

    For QPSK, divide each RE by its magnitude to form unit phasors and
    average their angles; returns phase to APPLY (multiply by exp(1j*theta)).
    """
    z = pbch_re.reshape(-1)
    mag = np.abs(z) + 1e-12
    u = z / mag
    avg = u.mean()
    return -np.angle(avg)


def apply_phase(pbch_re: np.ndarray, theta: float) -> np.ndarray:
    return pbch_re * np.exp(1j * theta)


def normalize_amplitude(pbch_re: np.ndarray) -> np.ndarray:
    """Normalize per-symbol RMS amplitude to 1 for a flat scalar equalizer."""
    y = pbch_re.copy()
    for i in range(y.shape[0]):
        rms = np.sqrt(np.mean(np.abs(y[i]) ** 2) + 1e-12)
        y[i] = y[i] / rms
    return y


def qpsk_llrs(pbch_eq: np.ndarray, noise_var: float = 1.0) -> np.ndarray:
    """Compute simple LLRs for QPSK symbols assuming unit-energy constellation.
    Returns LLRs array of shape (num_bits,), mapping Gray-coded QPSK.
    """
    s = pbch_eq.reshape(-1)
    # Bits: b0 from real, b1 from imag (Gray)
    llr0 = 2 * s.real / (noise_var + 1e-9)
    llr1 = 2 * s.imag / (noise_var + 1e-9)
    return np.column_stack([llr0, llr1]).reshape(-1)


_PBCH_INTERLEAVER_PERM = [
    1, 17, 9, 25, 5, 21, 13, 29,
    3, 19, 11, 27, 7, 23, 15, 31,
    0, 16, 8, 24, 4, 20, 12, 28,
    2, 18, 10, 26, 6, 22, 14, 30,
]


def _pbch_subblock_interleave(stream_index: int) -> np.ndarray:
    C = 32
    D = 40
    R = int(np.ceil(D / C))
    K_pi = R * C
    N_dummy = K_pi - D
    seq: List[Optional[Tuple[int, int]]] = [None] * N_dummy + [(stream_index, k) for k in range(D)]
    matrix = np.empty((R, C), dtype=object)
    idx = 0
    for r in range(R):
        for c in range(C):
            matrix[r, c] = seq[idx]
            idx += 1
    permuted = np.empty_like(matrix)
    for new_c, old_c in enumerate(_PBCH_INTERLEAVER_PERM):
        permuted[:, new_c] = matrix[:, old_c]
    v: List[Optional[Tuple[int, int]]] = []
    for c in range(C):
        for r in range(R):
            v.append(permuted[r, c])
    return np.array(v, dtype=object)


_PBCH_RATE_MATCH_MAP: Dict[int, List[Tuple[int, int]]] = {}
_PBCH_CRC_MASKS = {
    1: 0x0000,
    2: 0xFFFF,
    4: 0xAAAA,
}


def _pbch_rate_match_map(rv_idx: int) -> List[Tuple[int, int]]:
    rv_idx = int(rv_idx) & 0x3
    if rv_idx in _PBCH_RATE_MATCH_MAP:
        return _PBCH_RATE_MATCH_MAP[rv_idx]
    streams = [_pbch_subblock_interleave(i) for i in range(3)]
    w = np.concatenate(streams)
    N_cb = len(w)
    R_tc = int(np.ceil(40 / 32))
    ceil_term = int(np.ceil(N_cb / (8 * R_tc)))
    k0 = R_tc * (2 * ceil_term * rv_idx + 2)
    mapping: List[Tuple[int, int]] = []
    idx = k0
    while len(mapping) < 480:
        val = w[idx % N_cb]
        if val is not None:
            mapping.append(val)
        idx += 1
    _PBCH_RATE_MATCH_MAP[rv_idx] = mapping
    return mapping


# --- Viterbi decoder (rate-1/3, K=7) scaffold ---
def _build_trellis_rate13_k7():
    K = 7
    n_states = 1 << (K - 1)
    # LTE convolutional code (per typical references): octal generators g0=133, g1=171, g2=165
    gens = [0o133, 0o171, 0o165]
    next_state = np.zeros((n_states, 2), dtype=int)
    out_bits = np.zeros((n_states, 2, 3), dtype=int)
    for s in range(n_states):
        for b in (0, 1):
            inp = b
            reg = ((s << 1) | inp) & (n_states - 1)
            next_state[s, b] = reg
            # Compute 3 output bits
            for j, g in enumerate(gens):
                taps = reg | (inp << (K - 1))
                out = bin(taps & g).count("1") & 1
                out_bits[s, b, j] = out
    return next_state, out_bits


_NEXT_STATE, _OUT_BITS = _build_trellis_rate13_k7()


def _compute_prev_states() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev_states = np.zeros_like(_NEXT_STATE)
    prev_inputs = np.zeros_like(_NEXT_STATE, dtype=np.int8)
    prev_outputs = np.zeros((_NEXT_STATE.shape[0], 2, _OUT_BITS.shape[-1]), dtype=np.int8)
    counts = np.zeros(_NEXT_STATE.shape[0], dtype=int)
    for s in range(_NEXT_STATE.shape[0]):
        for b in (0, 1):
            ns = _NEXT_STATE[s, b]
            idx = counts[ns]
            prev_states[ns, idx] = s
            prev_inputs[ns, idx] = b
            prev_outputs[ns, idx] = _OUT_BITS[s, b]
            counts[ns] += 1
    return prev_states, prev_inputs, prev_outputs


_PREV_STATE, _PREV_INPUT, _PREV_OUTPUT = _compute_prev_states()
_PREV_SIGNS = 1 - 2 * _PREV_OUTPUT


def viterbi_decode_rate13_k7(llrs: np.ndarray) -> np.ndarray:
    """Soft-decision Viterbi for rate-1/3, K=7 convolutional code.
    llrs: length = 3 * N coded bits in order [c0,c1,c2, c0,c1,c2, ...]
    Returns estimated N information bits (tail-biting not handled).
    """
    next_state = _NEXT_STATE
    out_bits = _OUT_BITS
    n_states = next_state.shape[0]
    n_sym = llrs.size // 3
    # Path metrics (start from state 0)
    PM = np.full((n_sym + 1, n_states), np.inf)
    PM[0, 0] = 0.0
    prev = np.full((n_sym, n_states), -1, dtype=int)
    prev_b = np.full((n_sym, n_states), 0, dtype=np.int8)
    for t in range(n_sym):
        y = llrs[3 * t:3 * t + 3]
        for s in range(n_states):
            if np.isinf(PM[t, s]):
                continue
            for b in (0, 1):
                ns = next_state[s, b]
                c = out_bits[s, b]
                # Branch metric: -sum(LLR * (1-2*c))  (assuming LLR>0 means bit=0)
                bm = -np.sum(y * (1 - 2 * c))
                m = PM[t, s] + bm
                if m < PM[t + 1, ns]:
                    PM[t + 1, ns] = m
                    prev[t, ns] = s
                    prev_b[t, ns] = b
    # Traceback to state 0 (terminated) if available, else best state
    end_state = int(np.argmin(PM[n_sym]))
    bits = np.zeros(n_sym, dtype=np.uint8)
    s = end_state
    for t in range(n_sym - 1, -1, -1):
        b = prev_b[t, s]
        bits[t] = b
        s = prev[t, s]
        if s < 0:
            s = 0
    return bits


def viterbi_decode_rate13_k7_tailbiting(llrs: np.ndarray) -> np.ndarray:
    """Tail-biting Viterbi decoding for rate-1/3, K=7 convolutional code."""
    if llrs.size % 3 != 0 or llrs.size == 0:
        return np.array([], dtype=np.uint8)
    llrs = llrs.reshape(-1, 3)
    n_sym = llrs.shape[0]
    n_states = _NEXT_STATE.shape[0]
    pm = np.full((n_states, n_states), np.inf)
    pm[np.arange(n_states), np.arange(n_states)] = 0.0
    prev_state = np.full((n_sym, n_states, n_states), -1, dtype=np.int16)
    prev_input = np.zeros((n_sym, n_states, n_states), dtype=np.int8)
    for t in range(n_sym):
        y = llrs[t]
        bm = -np.einsum('j,sij->si', y, _PREV_SIGNS, optimize=True)
        cand0 = np.take(pm, _PREV_STATE[:, 0], axis=1) + bm[:, 0][None, :]
        cand1 = np.take(pm, _PREV_STATE[:, 1], axis=1) + bm[:, 1][None, :]
        choose0 = cand0 <= cand1
        pm = np.where(choose0, cand0, cand1)
        prev_state[t] = np.where(choose0, _PREV_STATE[:, 0][None, :], _PREV_STATE[:, 1][None, :])
        prev_input[t] = np.where(choose0, _PREV_INPUT[:, 0][None, :], _PREV_INPUT[:, 1][None, :])
    final_metrics = pm[np.arange(n_states), np.arange(n_states)]
    best_start = int(np.argmin(final_metrics))
    if np.isinf(final_metrics[best_start]):
        return np.array([], dtype=np.uint8)
    bits = np.zeros(n_sym, dtype=np.uint8)
    state = best_start
    for t in range(n_sym - 1, -1, -1):
        b = prev_input[t, best_start, state]
        bits[t] = b
        state = prev_state[t, best_start, state]
    return bits


def crc16_ccitt_bits(bits: np.ndarray, init: int = 0xFFFF) -> int:
    poly = 0x1021
    crc = init
    for b in bits.astype(int):
        crc ^= (b & 1) << 15
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) & 0xFFFF) ^ poly
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def pbch_descrambler(bits: np.ndarray, ncellid: int, i_mod4: int = 0) -> np.ndarray:
    seq = pbch_scrambling_sequence(ncellid, bits.size)
    return bits ^ seq


def pbch_descrambler_candidates(bits: np.ndarray, ncellid: int, i_mod4: int) -> Tuple[np.ndarray, ...]:
    return (pbch_descrambler(bits, ncellid, i_mod4),)


def _gold_seq(c_init: int, length: int) -> np.ndarray:
    """Generate LTE Gold sequence c(n) of given length (uint8 0/1)."""
    x1 = np.zeros(length + 1600 + 31, dtype=np.uint8)
    x2 = np.zeros_like(x1)
    x1[0] = 1
    for n in range(31, x1.size):
        x1[n] = (x1[n - 3] ^ x1[n - 31]) & 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for n in range(31, x2.size):
        x2[n] = (x2[n - 3] ^ x2[n - 2] ^ x2[n - 1] ^ x2[n - 31]) & 1
    return (x1[1600:1600 + length] ^ x2[1600:1600 + length]).astype(np.uint8)


def pbch_scrambling_sequence(ncellid: int, length: int, offset: int = 0) -> np.ndarray:
    seq = _gold_seq(ncellid, length + offset)
    return seq[offset:offset + length]


def pbch_scramble_seq(length: int, ncellid: int, i_mod4: int) -> np.ndarray:
    return pbch_scrambling_sequence(ncellid, length)


def descramble_llrs(llrs: np.ndarray, ncellid: int, i_mod4: int) -> np.ndarray:
    seq = pbch_scrambling_sequence(ncellid, llrs.size)
    signs = 1.0 - 2.0 * seq.astype(float)
    return llrs * signs


def pbch_scramble_seq_variants(length: int, ncellid: int, i_mod4: int) -> Tuple[np.ndarray, ...]:
    """Produce multiple plausible PBCH scrambling sequences for robustness."""
    c_inits = [
        ((ncellid & 0x3FF) << 9) | ((i_mod4 & 0x3) << 4) | 0x1FF,
        ((ncellid & 0x3FF) << 10) | ((i_mod4 & 0x3) << 5) | 0x155,
        ((ncellid & 0x3FF) << 7) | ((i_mod4 & 0x3) << 2) | 0x035,
        ((ncellid & 0x3FF) ^ ((i_mod4 & 0x3) << 9) ^ 0x3E1) & 0x7FFFFFFF,
    ]
    outs = []
    for ci in c_inits:
        outs.append(_gold_seq(ci, length))
    return tuple(outs)


def deratematch_pbch_llrs(llrs480: np.ndarray, i_mod4: int) -> Optional[np.ndarray]:
    """Invert PBCH rate matching (TS 36.212 5.1.4.2) to recover 120 soft bits."""
    if llrs480.size < 480:
        return None
    mapping = _pbch_rate_match_map(i_mod4)
    acc = np.zeros((3, 40), dtype=float)
    for llr, (stream, bit_idx) in zip(llrs480[:480], mapping):
        acc[stream, bit_idx] += float(llr)
    out = np.empty(120, dtype=float)
    idx = 0
    for bit_idx in range(40):
        for stream in range(3):
            out[idx] = acc[stream, bit_idx]
            idx += 1
    return out


def bits_to_uint(bits: np.ndarray) -> int:
    """Convert an array of bits (MSB-first) to integer."""
    val = 0
    for b in bits.astype(int):
        val = ((val << 1) | (b & 1)) & 0xFFFFFFFF
    return int(val)


def parse_mib_fields(payload24: np.ndarray, i_mod4: int) -> Dict[str, object]:
    """Parse 24-bit MIB payload into fields using common LTE mapping.

    Mapping (MSB-first assumption):
      - bits[0:3]: dl-Bandwidth (index → NDLRB)
      - bits[3:6]: phich-Config (duration:1, Ng:2)
      - bits[6:14]: SFN MSB (8 bits)
      - bits[14:24]: spare (ignored)
    NFrame reconstructed as (SFN_MSB8 << 2) | i_mod4.
    """
    b = payload24.astype(int)
    if b.size < 24:
        raise ValueError("MIB payload must be 24 bits")
    bw_idx = bits_to_uint(b[0:3])
    phich = bits_to_uint(b[3:6])
    sfn_msb8 = bits_to_uint(b[6:14])
    # dl-Bandwidth index mapping per LTE
    ndlrb_map = {0: 6, 1: 15, 2: 25, 3: 50, 4: 75, 5: 100}
    ndlrb = ndlrb_map.get(bw_idx, None)
    # PHICH: duration (1 bit) + Ng (2 bits)
    phich_duration = 'Normal' if ((phich >> 2) & 0x1) == 0 else 'Extended'
    ng_map = {0: '1/6', 1: '1/2', 2: '1', 3: '2'}
    ng = ng_map.get(phich & 0x3, None)
    nframe = ((sfn_msb8 & 0xFF) << 2) | (i_mod4 & 0x3)
    return {
        'NDLRB_from_MIB': ndlrb,
        'PHICHDuration': phich_duration,
        'Ng': ng,
        'NFrame': int(nframe),
    }


def try_decode_mib_from_pbch(pbch_re: np.ndarray, ncellid: int) -> Optional[Dict[str, object]]:
    """Placeholder MIB decoder: returns None by default.

    A full implementation should:
      - Remove CRS REs per 36.211 Table 6.10.1 and NCellID, CellRefP hypothesis
      - Equalize using CRS-based channel estimates
      - Descramble PBCH bits with cell-ID-specific sequence (36.211 6.6.1)
      - De-rate match and Viterbi-decode (36.212 5.3.1.1, K=7) to 40 bits (24 MIB + 16 CRC)
      - Verify CRC (scrambled with NCellID) and extract fields: NDLRB, PHICHDuration, Ng, SFN, CellRefP
    """
    for cellrefp_h in (1, 2, 4):
        mask = crs_data_mask_for_pbch(ncellid, cellrefp=cellrefp_h)
        data_re = pbch_re.transpose(1, 0)[mask.transpose(1, 0)].reshape(-1)
        llrs_all = qpsk_llrs(data_re, noise_var=1.0)
        if llrs_all.size < 480:
            continue
        max_start = llrs_all.size - 480
        for rv in range(4):
            for st in range(0, max_start + 1):
                llrs480 = descramble_llrs(llrs_all[st:st + 480], ncellid, rv)
                llrs120 = deratematch_pbch_llrs(llrs480, rv)
                if llrs120 is None:
                    continue
                decoded = viterbi_decode_rate13_k7_tailbiting(llrs120)
                if decoded.size < 40:
                    continue
                payload = decoded[:24]
                crc_bits = decoded[24:40]
                calc = crc16_ccitt_bits(payload)
                got = bits_to_uint(crc_bits)
                mask_val = _PBCH_CRC_MASKS[cellrefp_h]
                if (got ^ mask_val) != calc:
                    continue
                fields = parse_mib_fields(payload, rv)
                if fields.get('NDLRB_from_MIB') in (6, 15, 25, 50, 75, 100) and fields.get('Ng') in ('1/6', '1/2', '1', '2'):
                    return {
                        'CellRefP': cellrefp_h,
                        'PHICHDuration': fields.get('PHICHDuration'),
                        'Ng': fields.get('Ng'),
                        'NFrame': fields.get('NFrame'),
                        'NDLRB_from_MIB': fields.get('NDLRB_from_MIB'),
                    }
    return None


def brute_force_mib_from_pbch(pbch_re: np.ndarray, ncellid_hint: Optional[int] = None, nid2_hint: Optional[int] = None) -> Optional[Dict[str, object]]:
    """Brute-force PBCH MIB over NCellID to improve robustness with limited data.

    Tries a small neighborhood around hint first (if provided), then scans the set.
    Returns dictionary including 'NCellID' on success.
    """
    # Candidate order: hint, then all 0..503
    # Restrict by NID2 if provided: NCellID = 3*NID1 + NID2
    if nid2_hint is not None and int(nid2_hint) in (0, 1, 2):
        cand_ids = [3 * nid1 + int(nid2_hint) for nid1 in range(168)]
    else:
        cand_ids = list(range(504))
    if ncellid_hint is not None and 0 <= int(ncellid_hint) <= 503:
        h = int(ncellid_hint)
        # Prioritize a small neighborhood
        near = [((h + d) % 504) for d in range(-5, 6)]
        rest = [c for c in cand_ids if c not in near]
        cand_ids = near + rest
    # Reuse the single-id decoder; stop at first plausible hit
    for nid in cand_ids:
        mib = try_decode_mib_from_pbch(pbch_re, nid)
        if mib and mib.get('NDLRB_from_MIB') in (6, 15, 25, 50, 75, 100):
            mib = dict(mib)
            mib['NCellID'] = nid
            return mib
    return None
