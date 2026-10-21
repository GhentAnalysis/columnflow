# coding: utf-8

"""
Sanity check of the shifted histograms written by cf.MergeShiftedHistograms.

Asserts that the shift axis contains the nominal shift plus both directions of every expected shift
source (sources unknown to the requested TAFs are dropped silently by columnflow, so a missing source
would otherwise go unnoticed) and that every up and down variation differs from the nominal one (a
shift falling back to nominal would produce identical histograms).

Usage: python check_shift_hists.py HISTS_PICKLE SHIFT_SOURCES
"""

from __future__ import annotations

import sys
import pickle

import numpy as np


def error(msg: str) -> int:
    print(f"::error::{msg}")
    return 1


def main(path: str, sources: list[str]) -> int:
    with open(path, "rb") as f:
        h = pickle.load(f)

    # MergeShiftedHistograms stores a single histogram, or one per process
    hists = h if isinstance(h, dict) else {"all": h}
    for key, h in hists.items():
        print(f"histogram {key}:\n{h}")
        if "shift" not in h.axes.name:
            return error(f"histogram {key} has no shift axis, found axes {list(h.axes.name)}")

        shifts = set(h.axes["shift"])
        expected = {"nominal"} | {f"{source}_{d}" for source in sources for d in ["up", "down"]}
        if (missing := expected - shifts):
            return error(f"histogram {key} misses shifts {','.join(sorted(missing))}, found {','.join(sorted(shifts))}")

        nominal = h[{"shift": "nominal"}].values(flow=True)
        if not np.all(np.isfinite(nominal)) or nominal.sum() <= 0:
            return error(f"nominal histogram {key} is empty or has non-finite contents")

        for shift in sorted(expected - {"nominal"}):
            values = h[{"shift": shift}].values(flow=True)
            if not np.all(np.isfinite(values)):
                return error(f"histogram {key} has non-finite contents for shift {shift}")
            rel = values.sum() / nominal.sum() - 1
            print(f"{key}, {shift}: relative change of the integral {rel:+.5f}")
            if np.allclose(values, nominal, rtol=1e-7, atol=0):
                return error(f"histogram {key} for shift {shift} is identical to nominal")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2].split(",")))
