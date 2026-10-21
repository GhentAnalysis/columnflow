# coding: utf-8

"""
Sanity check of a datacard and its shapes file written by cf.CreateDatacards.

Asserts that the card declares the process and every shape parameter, and that the shapes file
contains the nominal and data_obs histograms plus up and down variations of every shape parameter,
all with positive integrals and with variations that differ from the nominal shape.

Usage: python check_datacard.py DATACARD SHAPES_ROOT CATEGORY PROCESS SHAPE_PARAMETERS
"""

from __future__ import annotations

import sys

import numpy as np
import uproot


def error(msg: str) -> int:
    print(f"::error::{msg}")
    return 1


def main(card_path: str, shapes_path: str, category: str, process: str, parameters: list[str]) -> int:
    with open(card_path) as f:
        card = f.read()
    print(card)

    # parse the card into its whitespace separated rows
    rows = {}
    for line in card.splitlines():
        parts = line.split()
        if parts and not parts[0].startswith(("#", "-")):
            rows.setdefault(parts[0], []).append(parts)

    if "shapes" not in rows:
        return error("datacard has no shapes line")
    if not any(process in row[1:] for row in rows.get("process", [])):
        return error(f"process '{process}' is not declared in the datacard")
    for param in parameters:
        if not any(len(row) > 1 and row[1] == "shape" for row in rows.get(param, [])):
            return error(f"shape parameter '{param}' is not declared in the datacard")

    with uproot.open(shapes_path) as f:
        keys = {key.split(";")[0] for key in f.keys()}
        print("shapes:", ", ".join(sorted(keys)))

        def values(name: str) -> np.ndarray | None:
            key = f"{category}/{name}"
            return f[key].values() if key in keys else None

        nominal = values(process)
        for name, vals in [(process, nominal), ("data_obs", values("data_obs"))]:
            if vals is None:
                return error(f"shapes file has no histogram {category}/{name}")
            if not np.all(np.isfinite(vals)) or vals.sum() <= 0:
                return error(f"histogram {category}/{name} is empty or has non-finite contents")

        for param in parameters:
            for direction in ["Up", "Down"]:
                name = f"{process}__{param}{direction}"
                vals = values(name)
                if vals is None:
                    return error(f"shapes file has no histogram {category}/{name}")
                if not np.all(np.isfinite(vals)) or vals.sum() <= 0:
                    return error(f"histogram {category}/{name} is empty or has non-finite contents")
                print(f"{name}: relative change of the integral {vals.sum() / nominal.sum() - 1:+.5f}")
                if np.allclose(vals, nominal, rtol=1e-7, atol=0):
                    return error(f"histogram {category}/{name} is identical to the nominal shape")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(",")))
