# coding: utf-8

"""
Sanity check of the selection stats written by cf.MergeSelectionStats.

cf.SelectEvents alone only proves that nothing crashed: an empty selection or a broken process_id
assignment still yields a valid (empty) plot and a green build downstream. This check asserts the
number of processed events and a selection efficiency band instead.

Usage: python check_stats.py STATS_JSON EXPECTED_NUM_EVENTS EFF_LOW EFF_HIGH
"""

from __future__ import annotations

import sys
import json


def main(path: str, expected_num_events: int, eff_low: float, eff_high: float) -> int:
    with open(path) as f:
        stats = json.load(f)

    num_events = stats["num_events"]
    num_events_selected = stats["num_events_selected"]
    efficiency = num_events_selected / num_events if num_events else 0.0

    print(f"num_events          : {num_events}")
    print(f"num_events_selected : {num_events_selected}")
    print(f"selection efficiency: {efficiency:.4f}")

    if num_events != expected_num_events:
        print(
            f"::error::num_events is {num_events}, expected {expected_num_events} - the CI test-data "
            "bundle or the dataset it is mapped to (tests/ci/ci_config_patch.py) changed without this "
            "assertion being updated",
        )
        return 1

    if not (eff_low <= efficiency <= eff_high):
        print(
            f"::error::selection efficiency {efficiency:.4f} is outside the expected [{eff_low}, {eff_high}] "
            "band - the selection, the test-data file, or a producer feeding it likely changed behaviour",
        )
        return 1

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])))
