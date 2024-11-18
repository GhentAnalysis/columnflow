# from collections import defaultdict
# from typing import Tuple

# import law

from columnflow.util import maybe_import, four_vec
from columnflow.columnar_util import set_ak_column
from columnflow.production import Producer, producer
# from columnflow.columnar_util_Ghent import TetraVec

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")


def _geometric_matching(particles1: ak.Array, particles2: ak.Array) -> (ak.Array, ak.Array):
    """
    Returns two awkward arrays.
    First contains that for each particle in **particles** the closest particle in the same event in **particles2**.
    Second tells you whether the found closest particle is contained within a cone of 0.2.
    """
    particles1, particles2 = ak.unzip(ak.cartesian([particles1, particles2], axis=1, nested=True))
    dr = particles1.delta_r(particles2)
    drmin_idx = ak.argmin(dr, axis=-1, keepdims=True)
    drmin = ak.flatten(dr[drmin_idx], axis=2)
    closest_match = ak.flatten(particles2[drmin_idx], axis=2)
    return closest_match, ak.fill_none(drmin < 0.2, False)


def unique(ak_array: ak.Array, axis=-1) -> ak.Array:

    ak_array1, ak_array2 = ak.unzip(ak.combinations(ak_array, 2, axis=1))

    mask = ak.all(ak_array1 != ak_array2, axis=-1)
    breakpoint()
    return mask


# map of the status flag name to the corresponding bit in statusFlags
_statusmap = ({
    "isPrompt": 0,
    "isDecayedLeptonHadron": 1,
    "isTauDecayProduct": 2,
    "isPromptTauDecayProduct": 3,
    "isDirectTauDecayProduct": 4,
    "isDirectPromptTauDecayProduct": 5,
    "isDirectHadronDecayProduct": 6,
    "isHardProcess": 7,
    "fromHardProcess": 8,
    "isHardProcessTauDecayProduct": 9,
    "isDirectHardProcessTauDecayProduct": 10,
    "fromHardProcessBeforeFSR": 11,
    "isFirstCopy": 12,
    "isLastCopy": 13,
    "isLastCopyBeforeFSR": 14,
})

# status flags that should be present for a prompt genparticle
_prompt_status = ["isPrompt", "isDirectPromptTauDecayProduct", "isHardProcess",
                 "fromHardProcess", "fromHardProcessBeforeFSR"]


@producer(
    uses=four_vec(
        ("Electron", "Muon"),
        ("pdgId", "genPartIdx")) |
    four_vec(
        ("GenPart"),
        ("pdgId", "status", "genPartIdxMother", "statusFlags"),
    ),
    produces=four_vec(
        {"Electron", "Muon"},
        {"isPrompt", "matchPdgId", "isChargeFlip", "isPhotonConversion"},
    ),
    mc_only=True,
    exposed=False,
)
def lepton_gen_features(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:

    genpart = events.GenPart

    for name, abs_pdgId in (("Electron", 11), ("Muon", 13)):

        lepton = events[name]

        # first check if already has a matched gen particle (include charge matching)
        is_nanoAOD_matched = (lepton.genPartIdx >= 0)
        is_nanoAOD_charge_matched = is_nanoAOD_matched & (lepton.pdgId == genpart.pdgId[lepton.genPartIdx])
        matched_genpart = genpart[lepton.genPartIdx]

        # if this fails apply geometric matching to stable leptons and photons

        # select stable gen particles
        stable_genpart = genpart[genpart.status == 1]

        # first look for closest mathing generator lepton within cone of 0.2
        gen_abs_pdgId = abs(stable_genpart.pdgId)
        geom_match_lepton, lepton_within_cone = _geometric_matching(lepton, stable_genpart[gen_abs_pdgId == abs_pdgId])

        # if not within cone of 0.2, allow for a photon match
        geom_match_photon, photon_within_cone = _geometric_matching(lepton, stable_genpart[gen_abs_pdgId == 22])

        # finally apply hierarchy to determine matched gen particle
        match = ak.Array(ak.zeros_like(geom_match_photon))
        match = ak.where(photon_within_cone, geom_match_photon, match)
        match = ak.where(lepton_within_cone, geom_match_lepton, match)
        match = ak.where(is_nanoAOD_charge_matched, matched_genpart, match)

        # check for matched gen particle if it fulfills all status flags for being prompt
        match_isPrompt = False
        for status in _prompt_status:
            match_isPrompt = match_isPrompt | (match.statusFlags & (1 << _statusmap[status]) != 0)

        matchMother = genpart[match.genPartIdxMother]
        count = 0
        while ak.any(matchMother.statusFlags & (1 << _statusmap["isLastCopy"]) == 0) & (count < 10):

            matchMother = ak.where(
                (matchMother.statusFlags & (1 << _statusmap["isLastCopy"]) == 0),
                genpart[matchMother.genPartIdxMother],
                matchMother
            )
            count += 1

        valid_match = is_nanoAOD_matched | lepton_within_cone | photon_within_cone
        is_chargeflip = (match.pdgId == -lepton.pdgId) & valid_match
        is_photonconversion = (
            (match.pdgId == -matchMother.pdgId) |
            (match.pdgId == 22) |
            ak.all(matchMother.pdgId == 23, axis=-1)
            # this line should be to check for (1) two match lepton have same genlevel lepton or (2) 2 leptons match to the same GenPart
        ) & valid_match

        events = set_ak_column(events, f"{name}.isPrompt", ak.fill_none(match_isPrompt, False, axis=-1))
        events = set_ak_column(events, f"{name}.matchPdgId", ak.fill_none(match.pdgId, -999, axis=-1))
        events = set_ak_column(events, f"{name}.isChargeFlip", ak.fill_none(is_chargeflip, False, axis=-1))
        events = set_ak_column(events, f"{name}.isPhotonConversion", ak.fill_none(is_photonconversion, False, axis=-1))

    return events
