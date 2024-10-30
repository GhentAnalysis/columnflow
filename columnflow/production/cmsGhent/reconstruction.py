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
from columnflow.columnar_util_Ghent import TetraVec

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
    produces=four_vec({"Zboson"}, {"isTrue"}),
)
def zboson_production(
    self: Producer,
    events: ak.Array,
    results: SelectionResult = None,
    nZ: int = 2,
    **kwargs,
) -> ak.Array:

    # produce events.Lepton
    events = self[lepton_production](events, results, **kwargs)
    lepton = events.Lepton

    # pad to 4 leptons per event to reconstruct 2 Z-bosons
    fill_with = {
        "pt": -999, "eta": -999, "phi": -999,
        "pdgId": -999, "mass": -999, "charge": -999
    }
    lepton = ak.fill_none(ak.pad_none(lepton, nZ * 2, axis=-1), fill_with)

    zboson_mem = []
    for _ in range(nZ):

        # save lepton index to find back the Z-boson leptons
        lepton = set_ak_column(lepton, "idx", ak.local_index(lepton, axis=1))

        _lep1, _lep2 = ak.unzip(ak.combinations(lepton, 2, axis=1))
        zboson_cands = TetraVec(_lep1) + TetraVec(_lep2)

        # mask for Z-boson criteria
        zcands_mask = (_lep1.charge != _lep2.charge) & (abs(_lep1.pdgId) ==
                    abs(_lep2.pdgId)) & (abs(ak.nan_to_num(zboson_cands.mass, nan=9999) - 91.1876) < 15)

        # Chi² to find the best combination of leptons for Z-boson reconstruction
        # weight up Chi² if it is not an OSSF pair so that OSSF pairs are always prioritized
        chi_sq = 9999 * ((_lep1.charge + _lep2.charge)**2 + (abs(_lep1.pdgId) -
                        abs(_lep2.pdgId))**2) + (ak.nan_to_num(zboson_cands.mass, nan=9999) - 91.1876)**2

        # find index of combinations that best match the z-boson, keep dimensionality to slice columns of the same dimension
        best_fit_idx = ak.argmin(chi_sq, axis=-1, keepdims=True)
        zboson = ak.flatten(zboson_cands[best_fit_idx], axis=-1)
        zboson_mask = ak.flatten(zcands_mask[best_fit_idx], axis=-1)

        # store the reconstructed zboson as a temporary column
        # should be optimized to not store in events but just an awkward array on its own
        events = set_ak_column(events, "temp.isTrue", ak.fill_none(zboson_mask, False))
        events = set_ak_column(events, "temp.idx", nZ * ak.local_index(events.temp, axis=0) + _)
        for var in ("pt", "eta", "phi", "mass"):
            var_func = getattr(zboson, var)
            events = set_ak_column(events, f"temp.{var}", ak.fill_none(ak.where(zboson_mask, var_func, 0.,), 0))
        zboson_mem.append(events.temp)

        # remove the leptons used to reconstruct the Z-boson above
        used_lepton_mask = (
            (lepton.idx != ak.flatten(_lep1.idx[best_fit_idx], axis=-1)) &
            (lepton.idx != ak.flatten(_lep2.idx[best_fit_idx], axis=-1))
        )

        lepton = lepton[used_lepton_mask]

    # concatenate all reconstructed zbosons in one long array and reshape it
    flat_zbosons = ak.flatten(ak.Array(zboson_mem))
    zbosons = ak.unflatten(flat_zbosons[ak.argsort(flat_zbosons.idx)], nZ)
    events = set_ak_column(events, "Zboson", zbosons[zbosons.isTrue])

    return events
