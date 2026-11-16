# coding: utf-8

"""
Generic hist hooks (change Ghent). Register them in the ``hist_hooks`` auxiliary of an analysis or config, e.g.

.. code-block:: python

    from columnflow.histogramming.hooks import mirror_shifts
    analysis_inst.x.setdefault("hist_hooks", {})["mirror_shifts"] = mirror_shifts

and invoke them with ``--hist-hooks <name>[,...]``.
"""

from __future__ import annotations

from collections import OrderedDict

import order as od

from columnflow.hist_util import add_missing_shifts
from columnflow.util import maybe_import
from columnflow.types import TYPE_CHECKING

if TYPE_CHECKING:
    hist = maybe_import("hist")
    from columnflow.tasks.framework.mixins import HistHookMixin


def mirror_shifts(
    task: HistHookMixin,
    hists: dict[od.Config, dict[od.Process, hist.Hist]],
    **kwargs,
) -> dict[od.Config, dict[od.Process, hist.Hist]]:
    """
    Builds the config shifts tagged ``mirrored`` from the other direction of their source: down = 2 · nominal - up (or
    up = 2 · nominal - down), per bin including the flow bins. The variance is copied from the filled direction (the
    mirror has the same statistical precision; the correlation with nominal is unknown).

    Use it for one-sided uncertainties: a shift tagged ``mirrored`` (e.g. ``<source>_down`` when only ``<source>_up`` is
    implemented) is accepted by :py:class:`~columnflow.tasks.framework.mixins.ShiftSourcesMixin` when the other direction
    is known, and skipped by :py:class:`~columnflow.tasks.histograms.MergeShiftedHistograms`, so no task fills it.
    This hook adds the bin when it is missing (the plot tasks add it filled with nominal before the hooks, datacard
    inputs lack it) and overwrites it. Processes without the filled direction or without ``nominal`` (e.g. data) are
    left as they are.

    Run it first, before hooks that combine processes or select or remove shifts: ``--hist-hooks mirror_shifts[,...]``.

    Edge cases: it needs the ``shift`` axis; only Weight and Double storages are handled; the mirrored content can be
    negative in low-statistics bins where up > 2 · nominal (not clipped).
    """
    _hists = OrderedDict()
    for config_inst, config_hists in hists.items():
        # mirrored shift -> filled shift of the same source
        mirror_map = {
            shift_inst.name: f"{shift_inst.source}_{od.Shift.UP if shift_inst.is_down else od.Shift.DOWN}"
            for shift_inst in config_inst.shifts
            if shift_inst.has_tag("mirrored")
        }

        _config_hists = OrderedDict()
        for proc, h in config_hists.items():
            shift_bins = list(h.axes["shift"])
            todo = {mirrored: filled for mirrored, filled in mirror_map.items() if filled in shift_bins}
            if not todo or "nominal" not in shift_bins:
                _config_hists[proc] = h
                continue

            # add missing mirrored bins (filled with nominal), then overwrite them in the full view
            h = h.copy()
            add_missing_shifts(h, set(todo), str_axis="shift", nominal_bin="nominal")
            shift_bins = list(h.axes["shift"])
            shift_axis = h.axes.name.index("shift")
            view = h.view(flow=True)
            index = lambda shift_name: (slice(None),) * shift_axis + (shift_bins.index(shift_name),)

            nominal = view[index("nominal")].copy()
            for mirrored, filled in todo.items():
                other = view[index(filled)]
                if view.dtype.names:
                    view.value[index(mirrored)] = 2 * nominal.value - other.value
                    view.variance[index(mirrored)] = other.variance
                else:
                    view[index(mirrored)] = 2 * nominal - other

            _config_hists[proc] = h

        _hists[config_inst] = _config_hists

    return _hists
