# coding: utf-8

"""
CI check for the datacard part of the collection shift test of
``.github/workflows/template_e2e.yaml``. Verifies that ``cf.CreateDatacards`` with the template's
``example`` inference model took the shapes of parameters contained in the collection shift
``ci_coll`` (see ``CI_COLLECTION`` in ``ci_config_patch.py``) from the collection histograms, next
to a regular shape parameter. Must run in a sandbox providing ``uproot`` (venv_columnar).

Usage: check_collection_shift_datacard.py <datacard.txt> <shapes.root>
"""

from __future__ import annotations

import sys

import numpy as np
import uproot

CATEGORY = "cat_incl"
PROCESS = "TT"
# shape parameters taken from the collection histograms (dynamic in the inference model)
COLLECTION_PARAMETERS = ["minbias_xs", "mur"]
# regular shape parameter with histograms of its own shifts
REGULAR_PARAMETERS = ["muf"]
# parameter added by the inference model to request the collection histograms, without effect
COLLECTION = "ci_coll"


def error(msg: str) -> None:
    print(f"::error::{msg}")
    sys.exit(1)


def main(card_path: str, shapes_path: str) -> None:
    with open(card_path) as f:
        card = f.read()
    print(card)

    # all shape parameters must be declared in the card
    shape_params = {
        line.split()[0]
        for line in card.splitlines()
        if len(line.split()) > 1 and line.split()[1] == "shape"
    }
    expected = set(COLLECTION_PARAMETERS + REGULAR_PARAMETERS + [COLLECTION])
    if missing := expected - shape_params:
        error(f"shape parameters {sorted(missing)} missing in datacard, found {sorted(shape_params)}")

    with uproot.open(shapes_path) as f:
        def values(name: str) -> np.ndarray:
            key = f"{CATEGORY}/{name}"
            if key not in f:
                error(f"histogram '{key}' missing in shapes file, available: {sorted(f.keys())}")
            return f[key].values()

        nominal = values(PROCESS)
        print(f"{PROCESS} nominal: sum {nominal.sum():.4f}")
        if not nominal.sum() > 0:
            error(f"nominal shape of process {PROCESS} is empty")

        # varied shapes of all parameters must exist and actually differ from nominal
        for param in COLLECTION_PARAMETERS + REGULAR_PARAMETERS:
            for direction in ["Up", "Down"]:
                vals = values(f"{PROCESS}__{param}{direction}")
                print(f"{PROCESS}__{param}{direction}: sum {vals.sum():.4f}")
                if np.allclose(vals, nominal, rtol=1e-6, atol=0):
                    error(f"shape of parameter '{param}' ({direction}) is identical to nominal")

        # the collection parameter only requests histograms, it must not have an effect itself
        for direction in ["Up", "Down"]:
            if not np.allclose(values(f"{PROCESS}__{COLLECTION}{direction}"), nominal):
                error(f"collection parameter '{COLLECTION}' ({direction}) unexpectedly differs from nominal")

    print("datacard with collection shifts is consistent")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        error(f"usage: {sys.argv[0]} <datacard.txt> <shapes.root>")
    main(*sys.argv[1:])
