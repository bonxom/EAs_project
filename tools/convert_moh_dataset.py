"""Explicit opt-in conversion of a trusted local upstream pickle to safe NPZ."""

import argparse
import pickle
from pathlib import Path

import numpy as np

from moh.problems.tsp_gls.dataset import validate_dataset_arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert trusted MoH pickle data. WARNING: pickle may execute code."
    )
    parser.add_argument(
        "--trusted-pickle",
        required=True,
        type=Path,
        help="Trusted local input only: loading pickle may execute code.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite")
    with args.trusted_pickle.open("rb") as handle:
        source = pickle.load(handle)
    if not isinstance(source, dict):
        parser.error("upstream dataset must be a dictionary")
    arrays = {
        target: np.asarray(source[original])
        for original, target in (
            ("coordinate", "coordinates"),
            ("distance_matrix", "distance_matrix"),
            ("cost", "cost"),
            ("optimal_tour", "optimal_tour"),
        )
    }
    validate_dataset_arrays(arrays, "converted")
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


if __name__ == "__main__":
    main()
