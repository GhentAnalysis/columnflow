# Columnflow Coding Invariants

## Column writes — always use set_ak_column

```python
# CORRECT — reassign events
from columnflow.columnar_util import set_ak_column
events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1), value_type=np.float32)

# WRONG — direct field mutation does not work
events["ht"] = ...   # wrong
events.ht    = ...   # wrong
```

## uses / produces — declare every column

- Every column read from `events` must be in `uses`.
- Every column written via `set_ak_column` must be in `produces`.
- Include sub-TAF objects in both sets to propagate their column declarations:

```python
@producer(uses={sub_producer, "extra"}, produces={sub_producer, "my_col"})
def parent(self, events, **kwargs):
    events = self[sub_producer](events, **kwargs)   # reassign!
```

## Imports — use maybe_import for heavy packages

```python
# At module level — deferred import (CORRECT)
from columnflow.util import maybe_import
ak = maybe_import("awkward")
np = maybe_import("numpy")

# coffea — inside function body only, never at module level
def my_func(self, events, **kwargs):
    import coffea.nanoevents.methods.vector
```

## Selectors — return signature and event mask

```python
def my_selector(self, events, stats, **kwargs) -> tuple[ak.Array, SelectionResult]:
    ...
    return events, SelectionResult(steps={"jet": mask})

# Exposed selector must set results.event:
from operator import and_
from functools import reduce
results.event = reduce(and_, results.steps.values())
```

## Monte Carlo guard

```python
if self.dataset_inst.is_mc:
    events = self[mc_weight](events, **kwargs)
```

## Vectorized operations — no Python loops over events

```python
# WRONG
for event in events: ...

# CORRECT
n_jet = ak.sum(events.Jet.pt > 25, axis=1)   # axis=1 = per-event
total = ak.sum(events.mc_weight)              # axis=0 = global scalar
```

## keep_columns — required for pre-Reduce columns

Columns produced in Calibrators/Selectors that are needed downstream must be listed:

```python
from columnflow.util import DotDict
from columnflow.columnar_util import ColumnCollection

cfg.x.keep_columns = DotDict.wrap({
    "cf.ReduceEvents": {
        "Jet.{pt,eta,phi,mass,btagDeepFlavB}",
        "Electron.{pt,eta,phi,mass,charge}",
        "MET.{pt,phi}",
        "event", "run", "luminosityBlock",
        ColumnCollection.ALL_FROM_SELECTOR,
    },
})
```

## TAF lifecycle hooks — when to use each

```python
@my_taf.init      # dynamic uses/produces/shifts; no task access
@my_taf.post_init # first hook with task access (for late registration)
@my_taf.requires  # add law task requirements (e.g. BundleExternalFiles)
@my_taf.setup     # load external resources (files, scale factors) onto self
@my_taf.teardown  # free memory
```

## Calling sub-TAFs

```python
events = self[other_producer](events, **kwargs)
events, sub_result = self[sub_selector](events, **kwargs)
```
