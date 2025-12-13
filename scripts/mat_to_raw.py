#!/usr/bin/env python3
"""
Convert a MATLAB .mat IQ capture into interleaved float32 .raw suitable for LTEIQ readers.
"""
import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.io import loadmat
try:
    import h5py  # type: ignore
except ImportError:  # h5py is optional unless v7.3 files are used
    h5py = None


def _to_complex(vec: np.ndarray) -> np.ndarray | None:
    """Return a 1D complex array from common MATLAB IQ layouts."""
    arr = np.asarray(vec).squeeze()
    if arr.size == 0:
        return None
    if np.iscomplexobj(arr):
        return arr.reshape(-1).astype(np.complex64)
    if arr.dtype.fields:
        lower_fields = {name.lower(): name for name in arr.dtype.fields}
        for r_name, i_name in (("real", "imag"), ("r", "i"), ("re", "im")):
            if r_name in lower_fields and i_name in lower_fields:
                real = arr[lower_fields[r_name]]
                imag = arr[lower_fields[i_name]]
                return (real + 1j * imag).astype(np.complex64).reshape(-1)
    if arr.ndim >= 2 and arr.shape[-1] == 2:
        pairs = arr.reshape(-1, 2)
        i, q = pairs[:, 0], pairs[:, 1]
        return (i + 1j * q).astype(np.complex64).reshape(-1)
    return None


def _priority(name: str) -> Tuple[int, str]:
    """Heuristic to prefer likely IQ variable names."""
    lname = name.lower()
    for score, token in enumerate(("iq", "iqdata", "samples", "data", "x")):
        if token in lname:
            return score, name
    return 99, name


def pick_iq_array(mat: Dict, preferred_key: str | None) -> Tuple[str | None, np.ndarray | None]:
    """Pick and convert an IQ-like variable from a loaded .mat dict."""
    user_keys = [preferred_key] if preferred_key else []
    all_keys = [k for k in mat.keys() if not k.startswith("__")]
    ordered = user_keys + [k for k in sorted(all_keys, key=_priority) if k not in user_keys]
    for key in ordered:
        if key not in mat:
            continue
        data = _to_complex(mat[key])
        if data is not None:
            return key, data
    return None, None


def _iter_h5_datasets(obj: "h5py.Group", prefix: str = "") -> Iterable[Tuple[str, "h5py.Dataset"]]:
    """Yield (path, dataset) for all datasets in an HDF5 tree."""
    for name, item in obj.items():
        if name.startswith("#"):  # skip h5py internals like #refs#
            continue
        full = f"{prefix}{name}"
        if isinstance(item, h5py.Dataset):
            yield full, item
        elif isinstance(item, h5py.Group):
            yield from _iter_h5_datasets(item, full + "/")


def _load_mat_hdf5(path: Path) -> Dict[str, np.ndarray]:
    """Load MATLAB v7.3 (HDF5) .mat into a dict of numpy arrays."""
    if h5py is None:
        raise RuntimeError("MATLAB v7.3 dosyaları için h5py gerekiyor. `pip install h5py` çalıştırın.")

    arrays: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for name, ds in _iter_h5_datasets(f):
            arrays[name] = np.asarray(ds[()])
    if not arrays:
        raise RuntimeError("HDF5 .mat içindeki veri kümeleri okunamadı.")
    return arrays


def load_mat_file(path: Path) -> Dict:
    """Load a .mat file (standard or v7.3) into a simple dict of arrays."""
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return _load_mat_hdf5(path)


def interleave_and_write(x: np.ndarray, output: Path) -> None:
    """Write interleaved float32 IQ (I0, Q0, I1, Q1, ...) to disk."""
    out = np.empty(x.size * 2, dtype=np.float32)
    out[0::2] = x.real.astype(np.float32)
    out[1::2] = x.imag.astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MATLAB .mat dosyasını interleaved float32 .raw formatına çevirir"
    )
    parser.add_argument("mat_file", type=Path, help="Girdi .mat dosyası")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Çıktı .raw yolu (varsayılan: giriş_adı.raw)",
    )
    parser.add_argument(
        "--key",
        help="Matlab değişken adı; verilmezse IQ benzeri ilk dizi otomatik seçilir",
    )
    args = parser.parse_args()

    if not args.mat_file.exists():
        parser.error(f".mat dosyası bulunamadı: {args.mat_file}")

    try:
        mat = load_mat_file(args.mat_file)
    except RuntimeError as exc:
        parser.error(str(exc))

    key, data = pick_iq_array(mat, args.key)

    if args.key and key is None:
        parser.error(f"Mat dosyasında '{args.key}' bulunamadı veya IQ formatında değil.")
    if data is None:
        parser.error("IQ verisine dönüştürülecek bir değişken bulunamadı; --key ile belirtmeyi deneyin.")

    out_path = args.output or args.mat_file.with_suffix(".raw")
    interleave_and_write(data, out_path)

    print(f"IQ değişkeni     : {key}")
    print(f"Örnek sayısı     : {data.size}")
    print(f"Çıktı            : {out_path}")
    print("Format           : interleaved float32 (I0, Q0, I1, Q1, ...)")


if __name__ == "__main__":
    main()
