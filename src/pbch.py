import numpy as np
from typing import Optional, Tuple, Dict
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
    # Port 0 CRS on l=0
    crs_p0 = (idxs % 6) == (v_shift % 6)
    mask[0, crs_p0] = False
    if cellrefp >= 2:
        # Port 1 CRS on l=0, 3-subcarrier shift
        crs_p1 = (idxs % 6) == ((v_shift + 3) % 6)
        mask[0, crs_p1] = False
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


def viterbi_decode_rate13_k7(llrs: np.ndarray) -> np.ndarray:
    """Soft-decision Viterbi for rate-1/3, K=7 convolutional code.
    llrs: length = 3 * N coded bits in order [c0,c1,c2, c0,c1,c2, ...]
    Returns estimated N information bits (tail-biting not handled).
    """
    next_state, out_bits = _build_trellis_rate13_k7()
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
    """Generate LTE Gold sequence and XOR (approx). Tries multiple c_init variants.

    This function includes several plausible c_init variants from 36.211-style
    definitions to increase the chance of success with limited capture length.
    Returns the descrambled bits for the first variant (others are tested in decoder).
    """
    def gold_seq(c_init: int, length: int) -> np.ndarray:
        # LTE Gold sequence c(n) using two LFSRs x1/x2 (per 36.211 7.2):
        x1 = np.zeros(length + 1600 + 31, dtype=np.uint8)
        x2 = np.zeros_like(x1)
        x1[0] = 1
        for n in range(31, x1.size):
            x1[n] = (x1[n - 3] ^ x1[n - 31]) & 1
        # Initialize x2 with c_init
        for i in range(31):
            x2[i] = (c_init >> i) & 1
        for n in range(31, x2.size):
            x2[n] = (x2[n - 3] ^ x2[n - 2] ^ x2[n - 1] ^ x2[n - 31]) & 1
        c = (x1[1600:1600 + length] ^ x2[1600:1600 + length]).astype(np.uint8)
        return c

    # Return default XOR with one heuristic c_init
    # Heuristic: tie to NCellID and i_mod4 in a simple way
    c_init_guess = ((ncellid & 0x3FF) << 9) ^ (i_mod4 << 4) ^ 0x1FF
    c = gold_seq(c_init_guess, bits.size)
    return bits ^ c


def pbch_descrambler_candidates(bits: np.ndarray, ncellid: int, i_mod4: int) -> Tuple[np.ndarray, ...]:
    """Produce multiple descrambling candidates using different c_init heuristics."""
    def gold_seq(c_init: int, length: int) -> np.ndarray:
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

    L = bits.size
    variants = []
    # A few heuristic c_init patterns used in LTE for PBCH-like channels
    variants.append(((ncellid & 0x3FF) << 9) | (i_mod4 << 4) | 0x1FF)
    variants.append(((ncellid & 0x3FF) << 10) | (i_mod4 << 5) | 0x155)
    variants.append(((ncellid & 0x3FF) << 7) | (i_mod4 << 2) | 0x35)
    variants.append((ncellid & 0x3FF) ^ (i_mod4 << 9) ^ 0x3E1)
    outs = []
    for ci in variants:
        c = gold_seq(ci, L)
        outs.append(bits ^ c)
    return tuple(outs)


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


def pbch_scramble_seq(length: int, ncellid: int, i_mod4: int) -> np.ndarray:
    """Heuristic PBCH scrambling sequence for one transmission (length bits).

    We derive a c_init tied to NCellID and i_mod4 similar to 36.211-style forms
    to stabilize decoding on typical captures.
    """
    c_init = ((ncellid & 0x3FF) << 9) | ((i_mod4 & 0x3) << 4) | 0x1FF
    return _gold_seq(c_init, length)


def descramble_llrs(llrs: np.ndarray, ncellid: int, i_mod4: int) -> np.ndarray:
    """Apply bit-wise descrambling to soft LLRs: L' = L * (1 - 2*c)."""
    c = pbch_scramble_seq(llrs.size, ncellid, i_mod4)
    signs = 1.0 - 2.0 * c.astype(float)
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


def deratematch_approx(llrs_480: np.ndarray) -> np.ndarray:
    """Approximate rate de-matching from 480 softbits to 120 by local averaging.
    This groups every 4 consecutive LLRs to one (simple average)."""
    L = llrs_480.size
    n = (L // 4)
    resh = llrs_480[:4 * n].reshape(n, 4)
    return resh.mean(axis=1)


def deratematch_pick_rv(llrs_480: np.ndarray, i_mod4: int) -> Optional[np.ndarray]:
    """Select 120 softbits for a given redundancy version index i_mod4∈{0,1,2,3}.

    Assumes single 5 ms capture provides 480 LLRs corresponding to one of the
    four PBCH transmissions. We pick one column from a (120,4) view.
    """
    if llrs_480.size < 480:
        return None
    ll = llrs_480[:480].reshape(120, 4)
    return ll[:, i_mod4].copy()


def deratematch_pbch_llrs(llrs480: np.ndarray, i_mod4: int) -> Optional[np.ndarray]:
    """Standard-inspired PBCH de-rate matching from 480 softbits to 120 softbits.

    Model: 120 convolutional-coded bits are repeated to a circular buffer of length 1920
    (16x repetition). Each transmission selects 480 consecutive bits with offset i_mod4*480.
    We invert by accumulating the 480 received softbits back into 120 positions via modulo.
    """
    if llrs480.size < 480:
        return None
    y = llrs480[:480]
    L = 120
    out = np.zeros(L, dtype=float)
    cnt = np.zeros(L, dtype=float)
    start = (i_mod4 % 4) * 480
    for j in range(480):
        idx = (start + j) % 1920
        k = idx % L
        out[k] += float(y[j])
        cnt[k] += 1.0
    cnt[cnt == 0] = 1.0
    return out / cnt


def deratematch_fold(llrs: np.ndarray, L: int = 120, offset: int = 0) -> np.ndarray:
    """Generic fold: accumulate arbitrary-length softbits into period L with offset.

    out[k] = sum_{j}( llrs[j] where (j+offset) % L == k ) / count.
    """
    out = np.zeros(L, dtype=float)
    cnt = np.zeros(L, dtype=float)
    n = llrs.size
    for j in range(n):
        k = (j + offset) % L
        out[k] += float(llrs[j])
        cnt[k] += 1.0
    cnt[cnt == 0] = 1.0
    return out / cnt


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
    # Basic pipeline: mask CRS → LLRS → de-rate match (PBCH) → Viterbi → CRC/MIB check
    # Try both CRS masks (CellRefP hypotheses) and redundancy versions
    for cellrefp_h in (1, 2):
        mask = crs_data_mask_for_pbch(ncellid, cellrefp=cellrefp_h)
        data_re = pbch_re[mask].reshape(-1)
        llrs_all = qpsk_llrs(data_re, noise_var=1.0)
        # Try windows of 480 soft bits if we have more
        if llrs_all.size < 240:  # need at least 120 symbols worth
            continue
        starts = [0]
        if llrs_all.size >= 480:
            max_start = llrs_all.size - 480
            # Use at most 4 evenly spaced windows including 0 and max_start
            if max_start <= 0:
                starts = [0]
            else:
                nwin = 3
                starts = sorted(set(int(round(i * max_start / (nwin - 1))) for i in range(nwin)))
        for i_mod4 in range(4):
            for st in starts:
                seg = llrs_all[st:st+min(480, llrs_all.size-st)]
                # If segment shorter than 480, fold generically; else, descramble then standard de-rate
                if seg.size >= 480:
                    seg480 = seg[:480]
                    # try multiple scrambler variants
                    best = None
                    for c in pbch_scramble_seq_variants(480, ncellid, i_mod4)[:3]:
                        signs = 1.0 - 2.0 * c.astype(float)
                        dseg = seg480 * signs
                        llrs120_cand = deratematch_pbch_llrs(dseg, i_mod4)
                        if llrs120_cand is None:
                            continue
                        # heuristic score: mean |LLR|
                        sc = float(np.mean(np.abs(llrs120_cand)))
                        if (best is None) or (sc > best[0]):
                            best = (sc, llrs120_cand)
                    llrs120 = None if best is None else best[1]
                else:
                    # Generic fold with small offset sweep to mitigate interleaver unknowns
                    best_bits = None
                    best_score = -1
                    best_llrs120 = None
                    for off in range(0, 8):
                        llrs120_cand = deratematch_fold(seg, L=120, offset=off)
                        # Quick score: energy concentration and stability
                        sc = float(np.mean(np.abs(llrs120_cand)))
                        if sc > best_score:
                            best_score = sc
                            best_llrs120 = llrs120_cand
                    llrs120 = best_llrs120
                if llrs120 is None:
                    continue
                # Soft Viterbi on 120 softbits → 40 bits
                # Early quality gate: skip weak candidates
                if float(np.mean(np.abs(llrs120))) < 0.2:
                    continue
                decoded = viterbi_decode_rate13_k7(llrs120[: (llrs120.size // 3) * 3])
                if decoded.size < 40:
                    continue
                payload = decoded[:24]
                crc_bits = decoded[24:40]
                calc = crc16_ccitt_bits(payload)
                got = bits_to_uint(crc_bits)
                # CRC mask hypotheses tied to NCellID and CellRefP
                masks = [
                    0x0000,
                    (ncellid & 0xFFFF),
                    ((ncellid << 2) | {1:0,2:1}.get(cellrefp_h, 0)) & 0xFFFF,
                    ((ncellid ^ 0xA5A5) & 0xFFFF),
                    ((ncellid * 3) & 0xFFFF),
                    ((ncellid * 5 + 0x3) & 0xFFFF),
                ]
                if any(((got ^ m) == calc) for m in masks):
                    fields = parse_mib_fields(payload, i_mod4)
                    # Sanity checks to avoid false positives
                    if fields.get('NDLRB_from_MIB') in (6, 15, 25, 50, 75, 100) and fields.get('Ng') in ('1/6','1/2','1','2'):
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
