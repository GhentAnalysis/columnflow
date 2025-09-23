# coding: utf-8

"""
Empty event weight producer.
"""

from columnflow.weight import WeightProducer, weight_producer
from columnflow.util import maybe_import
from columnflow.columnar_util import has_ak_column, Route

np = maybe_import("numpy")
ak = maybe_import("awkward")


@weight_producer
def empty(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    # simply return ones
    return events, ak.Array(np.ones(len(events), dtype=np.float32))

@weight_producer
def empty_mc(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    # simply return ones for data and mc_weight for mc
    weights = ak.Array(np.ones(len(events)))
    if self.dataset_inst.is_mc:
        if has_ak_column(events, "normalization_weight"):
                weights = weights * Route("normalization_weight").apply(events) 
        else:
            self.logger.warning_once(
                f"missing normalization_weight",
            )
    return events, weights

@empty_mc.init
def empty_mc_init(self: WeightProducer) -> None:
    if not getattr(self, "dataset_inst", None):
        return 
        
    if self.dataset_inst.is_mc :
        self.uses |= {Route("normalization_weight")}
