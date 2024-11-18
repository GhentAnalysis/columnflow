# coding: utf-8

"""
Definitions for reconstruction of physics objects
"""

import functools
import law

from columnflow.util import maybe_import, four_vec
from columnflow.selection import SelectionResult
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
    produces=four_vec({"Zboson"}),
)
def zboson_reconstruction(
    self: Producer,
    events: ak.Array,
    results: SelectionResult = None,
    nZ: int = 2,
    **kwargs,
) -> ak.Array:

    # produce events.Lepton
    events = self[lepton_production](events, results, **kwargs)
    lepton = events.Lepton

    # pad to nZ*2 leptons per event to reconstruct nZ Z-bosons
    fill_with = {
        "pt": -999, "eta": -999, "phi": -999,
        "pdgId": -999, "mass": -999, "charge": -999,
    }
    lepton = ak.fill_none(ak.pad_none(lepton, nZ * 2, axis=-1), fill_with)

    zboson_mem = []
    for i in range(nZ):

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
        events = set_ak_column(events, "temp.idx", nZ * ak.local_index(events.temp, axis=0) + i)
        for var in ("pt", "eta", "phi", "mass"):
            var_func = getattr(zboson, var)
            events = set_ak_column(events, f"temp.{var}", ak.fill_none(ak.where(zboson_mask, var_func, 0.,), 0))

        zboson_mem.append(events.temp)

        # remove the leptons used to reconstruct the Z-boson above
        unused_lepton_mask = (
            (lepton.idx != ak.flatten(_lep1.idx[best_fit_idx], axis=-1)) &
            (lepton.idx != ak.flatten(_lep2.idx[best_fit_idx], axis=-1))
        )
        lepton = lepton[unused_lepton_mask]

    # concatenate all reconstructed Z-bosons in one long array, re-arange it, reshape it, and filter the non-physical Z-bosons
    flat_zbosons = ak.concatenate(zboson_mem, axis=0)
    zbosons = ak.unflatten(flat_zbosons[ak.argsort(flat_zbosons.idx)], nZ)
    events = set_ak_column(events, "Zboson", zbosons[zbosons.isTrue])

    return events


@producer(
    uses={lepton_production} | four_vec({"MET"}),
    produces=four_vec({"Neutrino"}, {"isReal"}),
)
def neutrino_reconstruction(
    self: Producer,
    events: ak.Array,
    results: SelectionResult = None,
    w_lepton: ak.Array = None,
    **kwargs,
) -> ak.Array:
    """
    Producer to reconstruct a neutrino orignating from a leptonically decaying W boson.
    Assumes that Neutrino pt can be reconstructed via MET and that the W boson has been
    produced on-shell.

    TODO: reference
    """

    # TODO: might be outdated, should be defined in cmsdb
    w_mass = 80.379

    # transform MET into 4-vector
    events["MET"] = set_ak_column(events.MET, "mass", 0)
    events["MET"] = set_ak_column(events.MET, "eta", 0)
    events["MET"] = ak.with_name(events["MET"], "PtEtaPhiMLorentzVector")

    # produce events.Lepton
    if w_lepton is None:
        events = self[lepton_production](events, results, **kwargs)

        # pad to at least 1 lepton per event to reconstruct W-boson
        fill_with = {
            "pt": -999, "eta": -999, "phi": -999, "mass": -999,
        }
        w_lepton = ak.fill_none(ak.pad_none(events.Lepton, 1, axis=-1), fill_with)[:, 0]

    w_lepton = TetraVec(w_lepton)

    E_l = w_lepton.E
    pt_l = w_lepton.pt
    pz_l = w_lepton.pz
    pt_nu = events.MET.pt

    delta_phi = abs(w_lepton.delta_phi(events.MET))
    mu = w_mass**2 / 2 + pt_nu * pt_l * np.cos(delta_phi)

    # Neutrino pz will be calculated as: pz_nu = A +- sqrt(B-C)
    A = mu * pz_l / pt_l**2
    B = mu**2 * pz_l**2 / pt_l**4
    C = (E_l**2 * pt_nu**2 - mu**2) / pt_l**2

    pz_nu_1 = ak.where(
        B - C >= 0,
        # solution is real
        A + np.sqrt(B - C),
        # complex solution -> take only the real part
        A,
    )

    pz_nu_2 = ak.where(
        B - C >= 0,
        # solution is real
        A - np.sqrt(B - C),
        # complex solution -> take only the real part
        A,
    )

    pz_nu_solutions = [pz_nu_1, pz_nu_2]
    for i, pz_nu in enumerate(pz_nu_solutions, start=1):
        # convert to float64 to prevent rounding errors
        pt_nu = ak.values_astype(pt_nu, np.float64)
        pz_nu = ak.values_astype(pz_nu, np.float64)

        # calculate Neutrino eta to define the Neutrino 4-vector
        p_nu_1 = np.sqrt(pt_nu**2 + pz_nu**2)
        eta_nu_1 = np.log((p_nu_1 + pz_nu) / (p_nu_1 - pz_nu)) / 2
        # store Neutrino 4 vector components
        events[f"Neutrino{i}"] = events.MET
        events = set_ak_column(events, f"Neutrino{i}.eta", eta_nu_1)

        # sanity check: Neutrino pz should be the same as pz_nu within rounding errors
        sanity_check_1 = ak.sum(abs(events[f"Neutrino{i}"].pz - pz_nu) > abs(events[f"Neutrino{i}"].pz) / 100)
        if sanity_check_1:
            logger.warning(
                "Number of events with Neutrino.pz that differs from pz_nu by more than 1 percent: "
                f"{sanity_check_1} (solution {i})",
            )

        # sanity check: reconstructing W mass should always (if B-C>0) give the input W mass (80.4 GeV)
        W_on_shell = events[f"Neutrino{i}"] + w_lepton
        sanity_check_2 = ak.sum(abs(ak.where(B - C >= 0, W_on_shell.mass, w_mass) - w_mass) > 1)
        if sanity_check_2:
            logger.warning(
                "Number of events with W mass from reconstructed Neutrino (real solutions only) that "
                f"differs by more than 1 GeV from the input W mass: {sanity_check_2} (solution {i})",
            )

    # sanity check: for complex solutions, only the real part is considered -> both solutions should be identical
    sanity_check_3 = ak.sum(ak.where(B - C <= 0, events.Neutrino1.eta - events.Neutrino2.eta, 0))
    if sanity_check_3:
        raise Exception(
            "When finding complex neutrino solutions, both reconstructed Neutrinos should be identical",
        )

    # combine both Neutrino solutions by taking the solution with smaller absolute dR
    events = set_ak_column(
        events, "Neutrino",
        ak.where((events.Neutrino1.eta - w_lepton.eta)**2 + (events.Neutrino1.phi - w_lepton.phi)**2 > (events.Neutrino2.eta -
                 w_lepton.eta)**2 + (events.Neutrino2.phi - w_lepton.phi)**2, events.Neutrino2, events.Neutrino1),
    )
    events = set_ak_column(events, "Neutrino.isReal", B - C >= 0)

    return events


@ neutrino_reconstruction.init
def neutrino_reconstruction_init(self: Producer) -> None:
    # add variable instances to config
    add_neutrino_variables(self.config_inst)


@producer(
    uses={neutrino_reconstruction, lepton_production} | four_vec({"Electron", "Muon"}, {"pdgId", "charge"}),
    produces=four_vec({"Wboson"}, {"charge"}) | {neutrino_reconstruction},
)
def wz_reconstruction(
    self: Producer,
    events: ak.Array,
    results: SelectionResult = None,
    return_leptons=False,
    **kwargs,
):

    # produce events.Lepton
    events = self[lepton_production](events, results, **kwargs)
    lepton = events.Lepton

    # pad to 3 leptons per event to reconstruct W- and Z-bosons
    fill_with = {
        "pt": -999, "eta": -999, "phi": -999,
        "pdgId": -999, "mass": -999, "charge": -999,
    }
    lepton = ak.fill_none(ak.pad_none(lepton, 3, axis=-1), fill_with)

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

    # store the reconstructed zboson
    events = set_ak_column(events, "Zboson.isTrue", ak.fill_none(zboson_mask, False))
    for var in ("pt", "eta", "phi", "mass"):
        var_func = getattr(zboson, var)
        events = set_ak_column(events, f"Zboson.{var}", ak.fill_none(ak.where(zboson_mask, var_func, 0.,), 0))

    # remove the leptons used to reconstruct the Z-boson above
    unused_lepton_mask = (
        (lepton.idx != ak.flatten(_lep1.idx[best_fit_idx], axis=-1)) &
        (lepton.idx != ak.flatten(_lep2.idx[best_fit_idx], axis=-1))
    )

    w_lepton = lepton[unused_lepton_mask][:, 0]
    events = self[neutrino_reconstruction](events, results, lepton=w_lepton)

    wboson = TetraVec(w_lepton) + events.Neutrino
    for var in ("pt", "eta", "phi", "mass"):
        var_func = getattr(wboson, var)
        events = set_ak_column(events, f"Wboson.{var}", ak.fill_none(var_func, 0))
    events = set_ak_column(events, "Wboson.charge", w_lepton.charge)

    if return_leptons:
        return events, [lepton[~unused_lepton_mask], lepton[unused_lepton_mask]]

    return events
