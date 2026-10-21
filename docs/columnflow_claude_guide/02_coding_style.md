# 02 — Columnflow Coding Style

## Module Layout

A typical analysis module tree mirrors columnflow's own structure:

```text
myanalysis/
├── analysis/
│   └── my_analysis.py         # Analysis + Campaign + Config definitions
├── config/
│   ├── processes.py           # order.Process definitions
│   ├── datasets.py            # Campaign + DatasetInfo definitions
│   ├── variables.py           # order.Variable definitions
│   └── categories.py         # Category + Categorizer definitions
├── calibration/
│   └── default.py             # Calibrator definitions
├── selection/
│   ├── default.py             # Exposed Selector
│   ├── objects.py             # Internal object-selection Selectors
│   ├── trigger.py             # Trigger Selector
│   └── stats.py               # increment_stats Selector
├── production/
│   ├── default.py             # Main exposed Producer
│   ├── weights.py             # Weight Producers
│   └── features.py            # High-level variable Producers
├── categorization/
│   └── example.py             # Categorizer definitions
├── histogramming/
│   └── example.py             # HistProducer definitions
├── inference/
│   └── example.py             # InferenceModel definitions
└── tasks/
    └── ...                    # Custom law tasks
```

---

## Import Conventions

Always use `maybe_import` for packages not available in the default (non-columnar) sandbox. Only `numpy` and `awkward` are safe to import at module level via `maybe_import`. Never import `coffea` at module level.

```python
# Standard columnflow imports — safe at module level
from columnflow.production import Producer, producer
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.calibration import Calibrator, calibrator
from columnflow.util import maybe_import, four_vec, DotDict
from columnflow.columnar_util import set_ak_column, optional_column, has_ak_column, EMPTY_FLOAT, EMPTY_INT

# Deferred heavy imports — use maybe_import
np = maybe_import("numpy")
ak = maybe_import("awkward")

# coffea must ONLY be imported inside the function body, never at module level
def my_function(...):
    import coffea
    import coffea.nanoevents.methods.nanoaod
```

---

## TAF Decorator Pattern

All five TAF types share the same decorator pattern. The decorator registers `uses`, `produces`, and optional metadata on the class.

### Producer

```python
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")

@producer(
    uses={"Jet.pt", "Jet.eta"},       # columns to read from parquet
    produces={"ht", "n_jet"},          # columns to write to parquet
)
def my_producer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1), value_type=np.float32)
    events = set_ak_column(events, "n_jet", ak.sum(events.Jet.pt > 0, axis=1))
    return events
```

### Selector

```python
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import

ak = maybe_import("awkward")

@selector(
    uses={"Jet.pt", "Jet.eta"},
    produces=set(),
    exposed=True,                      # True = reachable from CLI --selector
)
def my_selector(
    self: Selector,
    events: ak.Array,
    stats: dict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    jet_mask = (events.Jet.pt > 25.0) & (abs(events.Jet.eta) < 2.4)
    jet_sel = ak.sum(jet_mask, axis=1) >= 2
    jet_indices = ak.local_index(events.Jet.pt)[jet_mask]

    return events, SelectionResult(
        steps={"jet": jet_sel},
        objects={"Jet": {"Jet": jet_indices}},
        aux={"jet_mask": jet_mask},
    )
```

### Calibrator

```python
from columnflow.calibration import Calibrator, calibrator
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")

@calibrator(
    uses={"Jet.pt"},
    produces={"Jet.pt"},
)
def my_calibrator(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    corrected_pt = events.Jet.pt * 1.02   # placeholder correction
    events = set_ak_column(events, "Jet.pt", corrected_pt)
    return events
```

---

## uses and produces Syntax

`uses` and `produces` are sets of strings, other TAF instances, or brace-expanded patterns.

```python
# Dot-notation for nested fields
uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass"}

# Brace expansion (bash-style) — equivalent to the above
uses={"Jet.{pt,eta,phi,mass}"}

# Wildcard — read all columns of a collection
uses={"Jet.*"}

# Include all uses/produces from another TAF (nested calls)
uses={other_producer}
produces={other_producer, "my_new_column"}

# four_vec helper — expands to pt, eta, phi, mass for each collection
from columnflow.util import four_vec
uses=four_vec({"Jet", "Electron"}, {"btagDeepFlavB"})
# equivalent to: Jet.{pt,eta,phi,mass,btagDeepFlavB}, Electron.{pt,eta,phi,mass}
```

---

## Nested TAF Calls

To call another TAF from within a TAF:

```python
events = self[other_producer](events, **kwargs)
events, sub_result = self[sub_selector](events, **kwargs)
```

The called TAF must be listed in the parent's `uses` and `produces` sets.

---

## SelectionResult Structure

```python
SelectionResult(
    steps={
        "step_name": bool_mask_1d,   # per-event boolean array
        "other_step": bool_mask_1d,
    },
    objects={
        "SourceField": {
            "DestField": index_array,  # ak.local_index(...)[mask]
        },
    },
    aux={
        "any_key": any_value,        # discarded after ReduceEvents
    },
)
```

The **exposed** selector must set `results.event` to the combined per-event boolean:

```python
from operator import and_
from functools import reduce

results.event = reduce(and_, results.steps.values())
```

Combine partial results with `+=`:

```python
results = SelectionResult()
events, jet_results = self[jet_selector](events, **kwargs)
results += jet_results
```

---

## TAF Lifecycle Hooks

Register additional behaviour on an existing TAF using decorator methods:

```python
@my_producer.pre_init
def my_producer_pre_init(self: Producer) -> None:
    # called before dependency tree; can set deps_kwargs
    ...

@my_producer.init
def my_producer_init(self: Producer) -> None:
    # dynamic uses/produces/shifts registration
    if self.dataset_inst.is_mc:
        self.uses.add("mc_weight")

@my_producer.post_init
def my_producer_post_init(self: Producer, task) -> None:
    # first hook with access to `task`
    ...

@my_producer.requires
def my_producer_requires(self: Producer, task, reqs: dict) -> None:
    # add extra law task requirements
    from columnflow.tasks.external import BundleExternalFiles
    reqs["ext_files"] = BundleExternalFiles.req(task)

@my_producer.setup
def my_producer_setup(self: Producer, task, reqs: dict, inputs: dict, reader_targets: dict) -> None:
    # load external resources (e.g. scale factor files) onto self
    bundle = inputs["ext_files"]
    self.sf_file = bundle["collection"][0]["muon_sf"].load()

@my_producer.teardown
def my_producer_teardown(self: Producer, task) -> None:
    # free memory
    del self.sf_file
```

---

## Accessing Config, Dataset, Analysis in a TAF

Inside any TAF function or hook, three objects are always available as `self` attributes:

```python
self.config_inst      # order.Config
self.dataset_inst     # order.Dataset
self.analysis_inst    # order.Analysis

# Common patterns
if self.dataset_inst.is_mc:
    ...
year = self.config_inst.campaign.x.year
lumi = self.config_inst.x.luminosity.nominal
```

---

## set_ak_column — The Only Way to Modify Events

Never modify `events` fields directly. Always use `set_ak_column` and reassign:

```python
from columnflow.columnar_util import set_ak_column
import numpy as np

# Scalar column
events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1), value_type=np.float32)

# Nested column
events = set_ak_column(events, "Jet.pt_corr", events.Jet.pt * 1.02)
```

---

## Coffea Behavior (Lorentz Vectors)

To perform four-vector arithmetic, attach coffea behaviour first:

```python
from columnflow.production.util import attach_coffea_behavior

events = self[attach_coffea_behavior](events, **kwargs)
# Now events.Jet, events.Electron etc. support .mass, .delta_r(), etc.

# Custom collection type
events = self[attach_coffea_behavior](
    events,
    collections={"MyJets": {"type_name": "Jet"}},
    **kwargs,
)
```

`attach_coffea_behavior` must be in `uses` of the TAF that calls it.

---

## Handling Missing / Optional Values

```python
from columnflow.columnar_util import EMPTY_FLOAT, EMPTY_INT, optional_column

# Access Jet.pt[:,2] safely (EMPTY_FLOAT for events with < 3 jets)
from columnflow.columnar_util import Route
jet3_pt = Route("Jet.pt[:,2]").apply(events, null_value=EMPTY_FLOAT)

# Pad to fixed length, fill with dict of defaults
padded = ak.fill_none(ak.pad_none(events.Jet, 4, axis=1), {"pt": -999.0, "eta": -999.0})

# Optional column in uses (present in some datasets only)
uses={optional_column("veto")}

# Check before using
from columnflow.columnar_util import has_ak_column
if has_ak_column(events, "veto"):
    events = events[~events.veto]
```

---

## DeferredColumn — Campaign-Dependent Column Requirements

```python
from columnflow.columnar_util import deferred_column

@deferred_column
def IF_RUN3(self, func) -> object:
    if func.config_inst.campaign.x.year >= 2022:
        return super(IF_RUN3, self).__call__(func)
    return None

@producer(
    uses={"common_col", IF_RUN3("run3_only_col")},
)
def my_prod(self, events, **kwargs):
    ...
```

---

## Naming Conventions

| Object | Convention | Example |
|---|---|---|
| TAF function | `snake_case` | `jet_selection`, `event_weights` |
| Exposed TAF | `snake_case` | `default` (standard name for the main TAF) |
| Column names | `snake_case` | `"ht"`, `"n_jet"`, `"Jet.pt"` |
| Config name | `snake_case` with year | `"run3_2022"`, `"l18"` (local 2018) |
| Version strings | `snake_case` or `v1` | `"dev1"`, `"prod1"`, `"selection_v2"` |
| Dataset names | `snake_case` with tag | `"tt_dl_powheg"`, `"data_mu_b"` |
| Process names | `snake_case` | `"tt"`, `"dy_lep"`, `"wjets"` |

---

## stats.json — Selection Statistics

The exposed Selector updates a `stats: dict` (or `defaultdict(float)`) in place. These keys are printed by `cf.SelectEvents`:

```python
stats["num_events"] += len(events)
stats["num_events_selected"] += ak.sum(results.event, axis=0)
stats["sum_mc_weight"] += ak.sum(events.mc_weight)
stats["sum_mc_weight_selected"] += ak.sum(events.mc_weight[results.event])
```

Use columnflow's built-in helper `increment_stats` where possible (from `columnflow.selection.stats`).

---

## Data vs. Monte Carlo Branching

Always guard MC-only operations:

```python
if self.dataset_inst.is_mc:
    events = self[mc_weight](events, **kwargs)
    events = self[pileup_weight](events, **kwargs)
```

For producers that should only run on MC, use the `init` hook to conditionally add columns:

```python
@my_producer.init
def my_producer_init(self: Producer) -> None:
    if self.dataset_inst.is_mc:
        self.uses.add("GenJet.pt")
        self.produces.add("n_gen_jet")
```
