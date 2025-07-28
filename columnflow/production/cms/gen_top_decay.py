# coding: utf-8

"""
Producers that determine the generator-level particles related to a top quark decay.
"""
from typing import Tuple

from __future__ import annotations

from columnflow.production import Producer, producer
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import, four_vec
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")

def _get_indexed_hard_children(gen_particle, start_index):
    children = gen_particle.distinctChildrenDeep
    children = set_ak_column(children, "index", gen_particle.distinctChildrenDeepIdxG - start_index)
    return children[children.hasFlags("isHardProcess")]


@producer(
    uses={"GenPart.{genPartIdxMother,pdgId,statusFlags}"},
    produces={"gen_top_decay"},
)
def gen_top_decay_products(self: Selector, events: ak.Array, **kwargs) -> Tuple[ak.Array, SelectionResult]:
    """
    Creates a new ragged column "gen_top_decay" with one element per hard top quark. Each element is
    a GenParticleArray with five or more objects in a distinct order: top quark, bottom quark,
    W boson, down-type quark or charged lepton, up-type quark or neutrino, and any additional decay
    produces of the W boson (if any, then most likly photon radiations). Per event, the structure
    will be similar to:

    .. code-block:: python

        [
            # event 1
            [
                # top 1
                [t1, b1, W1, q1/l, q2/n(, additional_w_decay_products)],
                # top 2
                [...],
            ],
            # event 2
            ...
        ]
    """

    genPart = events.GenPart
    start_index = np.cumsum(np.concatenate([0, ak.num(genPart)[:-1]]))
    genPart = set_ak_column(genPart, "index", ak.local_index(genPart.pdgId))
    print("genpartindex", genPart.index)

    # find hard top quarks
    top = genPart[abs(genPart.pdgId) == 6]
    top = top[top.hasFlags("isHardProcess")]
    top = top[~ak.is_none(top, axis=1)]
    print(top)
    top1 = top[:, 0]
    top2 = top[:, 1]
    
    top1_children = _get_indexed_hard_children(top1, start_index)
    top2_children = _get_indexed_hard_children(top2, start_index)

    b1 = top1_children[abs(top1_children.pdgId) == 5][:, 0]
    b2 = top2_children[abs(top2_children.pdgId) == 5][:, 0]

    w1 = top1_children[abs(top1_children.pdgId) == 24][:, 0]
    w2 = top2_children[abs(top2_children.pdgId) == 24][:, 0]

    w1_children = _get_indexed_hard_children(w1, start_index)
    w2_children = _get_indexed_hard_children(w2, start_index)

    w1up = w1_children[abs(w1_children.pdgId) % 2 == 1][:, 0]
    w2up = w2_children[abs(w2_children.pdgId) % 2 == 1][:, 0]

    w1dn = w1_children[abs(w1_children.pdgId) % 2 == 0][:, 0]
    w2dn = w2_children[abs(w2_children.pdgId) % 2 == 0][:, 0]
    
    indices = {
        "GenTop1": top1.index,
        "GenTop2": top2.index,
        "GenB1": b1.index,
        "GenB2": b2.index,
        "GenW1": w1.index,
        "GenW2": w2.index,
        "GenW1dec1": w1up.index,
        "GenW1dec2": w1dn.index,
        "GenW2dec1": w2up.index,
        "GenW2dec2": w2dn.index,
    }

    return events, SelectionResult(
        steps={},
        objects={
            "GenPart": {
                o: ak.fill_none(ak.from_regular(indices[o], axis=-1), 0) 
                for o in indices
            }
        },
    )

@gen_top_decay_products.skip
def gen_top_decay_products_skip(self: Producer, **kwargs) -> bool:
    """
    Custom skip function that checks whether the dataset is a MC simulation containing top
    quarks in the first place.
    """
    return self.dataset_inst.is_data or not self.dataset_inst.has_tag("has_top")
