---
paths:
  - "**/selection/**"
---

# Selector Rules

## Decorator pattern

```python
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import
ak = maybe_import("awkward")

@selector(
    uses={"Jet.{pt,eta}"},
    produces=set(),
    exposed=False,   # True only for the top-level CLI-reachable selector
)
def jet_selection(self: Selector, events: ak.Array, **kwargs) -> tuple[ak.Array, SelectionResult]:
    jet_mask = (events.Jet.pt > 25) & (abs(events.Jet.eta) < 2.4)
    jet_indices = ak.local_index(events.Jet.pt, axis=1)[jet_mask]
    return events, SelectionResult(
        steps={"jet": ak.sum(jet_mask, axis=1) >= 2},
        objects={"Jet": {"Jet": jet_indices}},
        aux={"jet_mask": jet_mask},   # discarded after ReduceEvents
    )
```

## SelectionResult fields

```python
SelectionResult(
    steps={
        "step_name": bool_1d_array,   # per-event; applied by ReduceEvents
    },
    objects={
        "SourceField": {
            "DestField": index_array,   # ak.local_index(...)[mask]
        },
    },
    aux={"key": value},   # temporary; not persisted after ReduceEvents
    event=combined_mask,  # required on the exposed selector's final result
)
```

## Composing sub-selectors

```python
results = SelectionResult()
events, jet_result = self[jet_selection](events, **kwargs)
results += jet_result
events, lep_result = self[lepton_selection](events, results, **kwargs)
results += lep_result

# Combine all steps into the final event mask (required for ReduceEvents)
from operator import and_
from functools import reduce
results.event = reduce(and_, results.steps.values())
```

## stats.json — increment in the exposed selector

```python
stats["num_events"]          += len(events)
stats["num_events_selected"] += ak.sum(results.event, axis=0)
if self.dataset_inst.is_mc:
    stats["sum_mc_weight"]          += ak.sum(events.mc_weight)
    stats["sum_mc_weight_selected"] += ak.sum(events.mc_weight[results.event])
```

Or use `from columnflow.selection.stats import increment_stats` (built-in helper).

## Accessing selected objects downstream within selection

```python
# Objects already reduced by results.objects from a prior sub-selector:
muon = events.Muon[results.objects.Muon.Muon]
jet  = events.Jet[results.objects.Jet.Jet]
```

## sorted_indices_from_mask helper

```python
from columnflow.columnar_util import sorted_indices_from_mask
# Sort selected objects by pt descending:
indices = sorted_indices_from_mask(object_mask, events.Jet.pt, ascending=False)
```
