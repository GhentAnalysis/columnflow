# coding: utf-8

"""
CI-only producers for the generated Ghent-template analysis, used by the collection shift test of
``.github/workflows/template_e2e.yaml``.

Like ``ci_config_patch.py``, this module lives outside of ``analysis_templates/`` so that it never
ships with a deployed analysis. The workflow copies it into the generated analysis package and
registers it via ``production_modules`` in ``law_user.cfg``.
"""

from __future__ import annotations

import importlib

from columnflow.production import Producer, producer
from columnflow.config_util import get_shifts_from_sources
from columnflow.util import maybe_import

ak = maybe_import("awkward")

# the template's default producer, imported relative to the analysis package this module is copied
# into, so that the analysis name is not hardcoded here
default = importlib.import_module(f"{__package__}.production.default").default


@producer(
    uses={default},
    produces={default},
)
def ci_jer_default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Wrapper around the template's default producer that additionally declares the jer shifts. No
    calibrator or selector of the template declares them, so cf.ReduceEvents resolves jer shifts to
    nominal while cf.ProduceColumns runs this producer at the jer shift itself (with Jet.pt/mass
    aliased to their jer variations) on the same nominal reduced events. This is exactly what the
    shifts_hist hist producer expects for "selection_dependent" subshifts of a collection shift.
    """
    return self[default](events, **kwargs)


@ci_jer_default.init
def ci_jer_default_init(self: Producer, **kwargs) -> None:
    if getattr(self, "dataset_inst", None) and self.dataset_inst.is_mc:
        self.shifts |= set(get_shifts_from_sources(self.config_inst, "jer"))
