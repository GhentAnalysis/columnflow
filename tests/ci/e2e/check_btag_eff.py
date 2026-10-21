# coding: utf-8

"""
Sanity check of the b-tag efficiency histogram written by cf.BTagEfficiency.

Asserts that efficiencies are within [0, 1], that every working point is present, that efficiencies
decrease from loose to tight working points, and that b jets are tagged more efficiently than light
jets. Bins without any jet yield NaN (0 / 0); the small CI test-data file leaves a few of them
empty (e.g. high-pt forward charm jets), so NaN is tolerated in at most MAX_NAN_FRACTION of the
bins and reported, while infinite values always fail.

Usage: python check_btag_eff.py BTAG_EFFICIENCY_PICKLE
"""

from __future__ import annotations

import sys
import pickle

import hist
import numpy as np


# maximum fraction of (empty, hence NaN) efficiency bins
MAX_NAN_FRACTION = 0.05


def error(msg: str) -> int:
    print(f"::error::{msg}")
    return 1


def main(path: str) -> int:
    with open(path, "rb") as f:
        h = pickle.load(f)

    print(h)
    names = list(h.axes.name)
    for axis in ["btag_wp", "systematic", "hadronFlavour"]:
        if axis not in names:
            return error(f"axis '{axis}' missing in b-tag efficiency histogram, found {names}")

    wps = list(h.axes["btag_wp"])
    if wps != ["L", "M", "T"]:
        return error(f"unexpected working points {wps}, expected ['L', 'M', 'T']")

    values = h.values()
    if np.any(np.isinf(values)):
        return error("b-tag efficiency histogram contains infinite values")
    nan_mask = np.isnan(values)
    if nan_mask.any():
        print(f"::warning::{nan_mask.sum()} of {nan_mask.size} b-tag efficiency bins are NaN (no jets)")
        for idx in np.argwhere(nan_mask):
            print(" - " + ", ".join(f"{ax.name}={ax.bin(i)}" for ax, i in zip(h.axes, idx)))
        if nan_mask.mean() > MAX_NAN_FRACTION:
            return error(f"more than {MAX_NAN_FRACTION:.0%} of the b-tag efficiency bins are NaN")
    finite = values[~nan_mask]
    if np.any(finite < 0) or np.any(finite > 1):
        return error(f"b-tag efficiencies outside [0, 1]: min {finite.min()}, max {finite.max()}")

    central = h[{"systematic": "central"}]

    def mean_eff(flavour: int, wp: str) -> float:
        # unweighted mean over the non-empty kinematic bins, enough to compare flavours and wps
        return float(np.nanmean(central[{"hadronFlavour": hist.loc(flavour), "btag_wp": wp}].values()))

    effs = {
        (flav, wp): mean_eff(flav, wp)
        for flav in [0, 5]
        for wp in wps
    }
    for (flav, wp), eff in effs.items():
        print(f"mean efficiency flavour {flav}, wp {wp}: {eff:.4f}")

    for flav in [0, 5]:
        if not (effs[(flav, "L")] >= effs[(flav, "M")] >= effs[(flav, "T")]):
            return error(f"efficiencies of flavour {flav} do not decrease from loose to tight working points")
    if not effs[(5, "M")] > effs[(0, "M")]:
        return error("medium working point efficiency of b jets is not larger than that of light jets")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
