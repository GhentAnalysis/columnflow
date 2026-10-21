# coding: utf-8

"""
Compares a weighted histogram (all_weights hist producer) to an unweighted one (cf_default hist
producer) of the same variable, both written by cf.MergeHistograms for the nominal shift.

Asserts that both are non-empty and have finite contents, that the unweighted one is filled with unit
weights and that the event weights actually changed the sum of weights.

Usage: python check_weighted_hists.py WEIGHTED_PICKLE UNWEIGHTED_PICKLE
"""

from __future__ import annotations

import sys
import pickle

import numpy as np


def error(msg: str) -> int:
    print(f"::error::{msg}")
    return 1


def load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(weighted_path: str, unweighted_path: str) -> int:
    weighted = load(weighted_path)
    unweighted = load(unweighted_path)

    sums = {}
    for name, h in [("weighted", weighted), ("unweighted", unweighted)]:
        print(f"{name} histogram:\n{h}")
        values = h.values(flow=True)
        if not np.all(np.isfinite(values)):
            return error(f"{name} histogram contains non-finite values")
        sums[name] = float(h.sum(flow=True).value)
        print(f"{name} sum of weights: {sums[name]:.4f}")
        if sums[name] <= 0:
            return error(f"{name} histogram is empty or has a non-positive sum of weights")

    # an unweighted histogram is filled with weight 1, so its sums of weights and of squared weights
    # are both its number of entries
    n_unweighted = float(unweighted.sum(flow=True).variance)
    if not np.isclose(n_unweighted, sums["unweighted"]):
        return error(
            f"unweighted histogram has sum of weights {sums['unweighted']} but sum of squared weights "
            f"{n_unweighted}, so it is not filled with unit weights",
        )

    if np.isclose(sums["weighted"], sums["unweighted"], rtol=1e-3):
        return error(
            "weighted and unweighted histograms have the same sum of weights, so the event weights of "
            "the all_weights hist producer were not applied",
        )

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
