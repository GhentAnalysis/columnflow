# coding: utf-8

"""
CI check for the collection shift test of ``.github/workflows/template_e2e.yaml``. Compares the
merged ``shifts_hist`` histograms of the collection shift (see ``CI_COLLECTION`` in
``ci_config_patch.py``) with histograms of the same hist producer created the standard way, one
shift per task. Must run in a sandbox providing ``hist`` (venv_columnar).

Usage: check_collection_shift_hists.py <collection_up.pickle> <minbias_xs_up.pickle> <nominal.pickle>
"""

from __future__ import annotations

import sys
import pickle

import numpy as np

# subshifts expected in the histogram of the collection shift ci_coll_up
EXPECTED_SUBSHIFTS = {"minbias_xs_up", "mur_up", "jer_up"}


def error(msg: str) -> None:
    print(f"::error::{msg}")
    sys.exit(1)


def load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def values(h, shift: str) -> np.ndarray:
    # reduce everything but the variable axis
    return h[{"category": sum, "process": sum, "shift": shift}].values()


def main(coll_path: str, minbias_path: str, nominal_path: str) -> None:
    h_coll = load(coll_path)
    h_minbias = load(minbias_path)
    h_nominal = load(nominal_path)

    # the collection histogram must contain exactly the subshifts, and not the collection itself
    shifts = set(h_coll.axes["shift"])
    print(f"shifts in collection histogram: {sorted(shifts)}")
    if shifts != EXPECTED_SUBSHIFTS:
        error(
            f"shift axis of the collection histogram is {sorted(shifts)}, expected "
            f"{sorted(EXPECTED_SUBSHIFTS)}; either the collection was resolved to nominal (not "
            "declared by shifts_hist) or subshifts were dropped",
        )

    nominal = values(h_nominal, "nominal")
    for shift in sorted(EXPECTED_SUBSHIFTS):
        vals = values(h_coll, shift)
        print(f"{shift:<14}: sum {vals.sum():.4f} (nominal {nominal.sum():.4f})")
        if not vals.sum() > 0:
            error(f"histogram for subshift '{shift}' is empty")

    # the weight-only subshift computed within the collection (nominal weight times variation over
    # nominal) must agree with the same shift created on its own (weight columns aliased)
    coll_minbias = values(h_coll, "minbias_xs_up")
    ref_minbias = values(h_minbias, "minbias_xs_up")
    if not np.allclose(coll_minbias, ref_minbias, rtol=1e-4, atol=1e-6):
        error(
            "subshift 'minbias_xs_up' of the collection differs from the standalone minbias_xs_up "
            f"histogram, max abs diff {np.max(np.abs(coll_minbias - ref_minbias)):.6g}",
        )

    # weight and selection dependent subshifts must actually change the histogram w.r.t. nominal,
    # otherwise weights or column aliases were not applied
    for shift in ["mur_up", "jer_up"]:
        if np.allclose(values(h_coll, shift), nominal, rtol=1e-6, atol=0):
            error(f"subshift '{shift}' is identical to nominal, the variation was not applied")

    print("collection shift histograms are consistent")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        error(f"usage: {sys.argv[0]} <collection_up.pickle> <minbias_xs_up.pickle> <nominal.pickle>")
    main(*sys.argv[1:])
