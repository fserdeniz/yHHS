#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Proje kök dizinini PYTHONPATH'e ekle
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lte_params import read_iq_file, analyze_lte_iq, LTEConfig, pretty_print_results


def main():
    p = argparse.ArgumentParser(description="Analyze LTE IQ capture (5 ms, 10 MHz)")
    p.add_argument("input", type=Path, help="Path to LTEIQ.raw file")
    args = p.parse_args()

    x = read_iq_file(str(args.input))
    cfg = LTEConfig()
    res = analyze_lte_iq(x, cfg)
    print(pretty_print_results(res))


if __name__ == "__main__":
    main()
