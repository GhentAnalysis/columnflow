# coding: utf-8

"""
Definitions for reconstruction of physics objects
"""

import functools
import law

from columnflow.util import maybe_import, four_vec
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production import Producer, producer
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import set_ak_column

from ttz.config.variables import add_neutrino_variables
from ttz.production.prepare_objects import prepare_objects
ak = maybe_import("awkward")
np = maybe_import("numpy")

logger = law.logger.get_logger(__name__)


@producer(
    uses=four_vec({"Electron", "Muon"},) | {attach_coffea_behavior},
    produces=four_vec({"Lepton"}),
)
def lepton_production(
    self: Producer,
    events: ak.Array,
    results: SelectionResult = None,
    **kwargs,
) -> ak.Array:

    if hasattr(events, "Lepton"):
        return events

    # initialize muon and electron columns
    muon = events.Muon
    electron = events.Electron

    # if results was given, filter the leptons
    if results:
        muon = muon[results.objects.Muon.Muon]
        electron = electron[results.objects.Electron.Electron]

    # add Lepton to events with coffea behaviour of muons
    lepton = ak.concatenate([muon, electron], axis=-1)
    lepton = lepton[ak.argsort(lepton.pt, axis=-1, ascending=False)]

    events = set_ak_column(events, "Lepton", lepton)

    lepton_coffea_collection = {
        "Lepton": {
            "type_name": "Muon",
            "check_attr": "metric_table",
            "skip_fields": "*Idx*G",
        },
    }
    events = self[attach_coffea_behavior](events, collections=lepton_coffea_collection, **kwargs)

    return events


@producer(
    uses={lepton_production} | four_vec({"Electron", "Muon"}, {"pdgId", "charge"}),
    produces=four_vec({"Zboson"}),
)
def zboson_production(
    self: Producer,
    events: ak.Array,
    results: SelectionResult = None,
    **kwargs,
) -> ak.Array:

    events = self[lepton_production](events, results, **kwargs)

    lepton = events.Lepton
