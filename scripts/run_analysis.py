#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Proje kök dizinini PYTHONPATH'e ekle
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lte_params import load_iq_auto_with_meta, analyze_lte_iq, pretty_print_results


def main():
    p = argparse.ArgumentParser(description="Analyze LTE IQ capture (auto LTE bandwidth, ~5 ms)")
    p.add_argument("input", type=Path, help="Path to input IQ file (.raw/.iq/.mat)")
    p.add_argument(
        "--key",
        help="MATLAB variable name to read IQ from when input is .mat (optional)",
    )
    p.add_argument(
        "--no-bruteforce",
        action="store_true",
        help="Skip PBCH brute-force search over NCellID (faster, MATLAB-like).",
    )
    p.add_argument(
        "--bruteforce-limit",
        type=int,
        default=60,
        help="Max NCellID candidates when brute-force is enabled (set 0 for unlimited).",
    )
    args = p.parse_args()

    x, meta = load_iq_auto_with_meta(str(args.input), mat_key=args.key)
    res = analyze_lte_iq(
        x,
        config=None,  # auto-select bandwidth unless user supplies
        enable_bruteforce=not args.no_bruteforce,
        bruteforce_limit=args.bruteforce_limit,
        fs_hint=meta.get('fs_hint') if meta else None,
    )
    print(pretty_print_results(res))


if __name__ == "__main__":
    main()
