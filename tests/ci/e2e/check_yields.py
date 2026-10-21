# coding: utf-8

"""
Sanity check of the yields written by cf.CreateYieldTable.

Asserts that every requested category has a positive, finite yield for the process, and that the
yields of the lepton channel categories (all categories but the first) are consistent with the yield
of the inclusive category (the first one): the channels are disjoint, so their sum must not exceed
the inclusive yield, and since they are defined on the reduced lepton collections (which can contain
more leptons than required by the selection), they need not cover it completely either, but must
cover at least MIN_CHANNEL_COVERAGE of it.

Usage: python check_yields.py YIELDS_JSON PROCESS CATEGORIES
"""

from __future__ import annotations

import sys
import json
import math


# minimum fraction of the inclusive yield covered by the channel categories (~0.987 observed)
MIN_CHANNEL_COVERAGE = 0.9


def error(msg: str) -> int:
    print(f"::error::{msg}")
    return 1


def main(path: str, process: str, categories: list[str]) -> int:
    with open(path) as f:
        yields = json.load(f)
    print(json.dumps(yields, indent=4))

    values = {}
    for category in categories:
        value = yields.get(category, {}).get(process)
        if value is None:
            return error(f"no yield for process '{process}' in category '{category}'")
        if not math.isfinite(value) or value <= 0:
            return error(f"yield {value} of process '{process}' in category '{category}' is not positive")
        values[category] = value

    incl, *channels = categories
    channel_sum = sum(values[category] for category in channels)
    coverage = channel_sum / values[incl]
    print(f"categories {','.join(channels)} cover {coverage:.4f} of category '{incl}'")
    if coverage > 1 + 1e-6:
        return error(
            f"yields of the disjoint categories {','.join(channels)} add up to {channel_sum}, more than "
            f"the yield {values[incl]} of category '{incl}'",
        )
    if coverage < MIN_CHANNEL_COVERAGE:
        return error(
            f"categories {','.join(channels)} cover only {coverage:.4f} of category '{incl}', expected at "
            f"least {MIN_CHANNEL_COVERAGE}",
        )

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3].split(",")))
