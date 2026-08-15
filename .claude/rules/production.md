---
paths:
  - "**/production/**"
---

# Producer Rules

## Decorator pattern

```python
from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.util import maybe_import, four_vec
ak = maybe_import("awkward")
np = maybe_import("numpy")

@producer(
    uses={"Jet.{pt,eta}", "Electron.{pt,eta,phi,mass}"},
    produces={"ht", "n_jet", "jet1_pt"},
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1), value_type=np.float32)
    events = set_ak_column(events, "n_jet", ak.num(events.Jet.pt, axis=1))
    from columnflow.columnar_util import Route
    events = set_ak_column(
        events, "jet1_pt",
        Route("Jet.pt[:,0]").apply(events, null_value=EMPTY_FLOAT),
        value_type=np.float32,
    )
    return events
```

## four_vec helper

```python
from columnflow.util import four_vec

# Expands to Collection.{pt,eta,phi,mass} (+ extra fields) for each collection:
uses=four_vec({"Jet", "Electron"}, {"btagDeepFlavB"})
# equivalent to: {"Jet.{pt,eta,phi,mass,btagDeepFlavB}", "Electron.{pt,eta,phi,mass}"}
```

## Nesting producers

```python
@producer(
    uses={sub_producer, "extra_col"},    # sub_producer's uses are inherited
    produces={sub_producer, "my_col"},   # sub_producer's produces are inherited
)
def parent(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[sub_producer](events, **kwargs)
    events = set_ak_column(events, "my_col", ...)
    return events
```

## MC-conditional columns — use init hook

```python
@my_producer.init
def my_producer_init(self: Producer) -> None:
    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return
    self.uses.add("mc_weight")
    self.produces.add("event_weight")

def my_producer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = set_ak_column(events, "event_weight", events.mc_weight * ...)
    return events
```

## Loading external files (scale factors, corrections)

```python
@sf_producer.requires
def sf_producer_requires(self, task, reqs):
    from columnflow.tasks.external import BundleExternalFiles
    reqs["ext"] = BundleExternalFiles.req(task)

@sf_producer.setup
def sf_producer_setup(self, task, reqs, inputs, reader_targets):
    bundle = inputs["ext"]["collection"][0]
    self.sf_file = bundle["muon_sf"].load()   # store on self for use in __call__
```

## Common column patterns

```python
# Scalar sum of jet pT
events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1), value_type=np.float32)

# Count objects
events = set_ak_column(events, "n_jet", ak.num(events.Jet.pt, axis=1))

# Leading object pT with EMPTY_FLOAT for events with no objects
from columnflow.columnar_util import Route, EMPTY_FLOAT
events = set_ak_column(events, "jet1_pt",
    Route("Jet.pt[:,0]").apply(events, null_value=EMPTY_FLOAT), value_type=np.float32)

# Invariant mass (requires attach_coffea_behavior first)
from columnflow.production.util import attach_coffea_behavior
events = self[attach_coffea_behavior](events, **kwargs)
mll = (events.Lepton[:, 0] + events.Lepton[:, 1]).mass
```
