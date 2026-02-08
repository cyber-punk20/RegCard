#!/usr/bin/env python3
"""
Filter a train-format CSV (tables#joins#predicates#cardinality) and drop all lines
where cardinality is 0 or empty.
"""

import argparse
import csv
from pathlib import Path


def _keep_row(parts):
    """True if row should be kept (cardinality not 0)."""
    card = parts[-1].strip() if len(parts) >= 4 else ""
    if card == "":
        return True  # empty = keep (e.g. unfilled)
    try:
        return int(card) != 0
    except ValueError:
        return True


def filter_zero_cardinality(input_path, output_path=None):
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path
    in_place = output_path.resolve() == input_path.resolve()

    with open(input_path, "r") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    kept = []
    for line in lines:
        parts = line.split("#")
        if _keep_row(parts):
            kept.append(line)

    out_file = output_path
    with open(out_file, "w", newline="") as f:
        for r in kept:
            f.write(r + "\n")

    return len(kept), len(lines) - len(kept)


def main():
    ap = argparse.ArgumentParser(description="Filter out rows with cardinality 0 from a train-format CSV")
    ap.add_argument("csv", type=str, help="Input CSV (delimiter #, last column = cardinality)")
    ap.add_argument("-o", "--output", type=str, default=None, help="Output CSV (default: overwrite input)")
    args = ap.parse_args()
    kept, dropped = filter_zero_cardinality(args.csv, args.output)
    print(f"Kept {kept} rows, dropped {dropped} rows (cardinality 0)")


if __name__ == "__main__":
    main()
