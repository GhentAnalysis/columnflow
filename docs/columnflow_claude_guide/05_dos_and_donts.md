# 05 — Do's and Don'ts for Columnflow Users

This is a practical checklist for anyone writing code with columnflow, covering the most common mistakes and best practices. Each rule has a concrete code example showing the wrong and correct approach.

---

## Event Data: the `events` Array

### DO use `set_ak_column` to add or modify columns

`events` is an awkward array. Its fields cannot be set by direct attribute assignment. Always use `set_ak_column` and **reassign** the return value.

```python
# CORRECT
from columnflow.columnar_util import set_ak_column
events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1), value_type=np.float32)

# WRONG — silently ignored or raises
events["ht"] = ak.sum(events.Jet.pt, axis=1)
events.ht    = ak.sum(events.Jet.pt, axis=1)
```

### DO reassign `events` after every TAF call

```python
# CORRECT
events = self[other_producer](events, **kwargs)
events = set_ak_column(events, "my_col", value)

# WRONG — the return value is discarded
self[other_producer](events, **kwargs)
```

### DON'T loop over events in Python

Columnflow is columnar: all operations must be vectorized with `ak.*`. Python loops over events are thousands of times slower and defeat the purpose of the framework.

```python
# WRONG — never do this
hts = []
for event in events:
    hts.append(sum(jet.pt for jet in event.Jet))

# CORRECT
ht = ak.sum(events.Jet.pt, axis=1)
```

### DO specify `axis=1` when reducing over objects within an event

```python
# WRONG — sums over everything (axis=None by default in some contexts)
n_jet = ak.sum(events.Jet.pt > 0)       # scalar, not per-event

# CORRECT — one value per event
n_jet = ak.sum(events.Jet.pt > 0, axis=1)   # shape (N_events,)
```

---

## TAF Declarations (`uses` and `produces`)

### DO declare every column you read in `uses`

Columnflow only loads columns from disk that are listed in `uses`. Missing a column means `events.Jet.pt` raises `AttributeError` at runtime.

```python
# CORRECT
@producer(
    uses={"Jet.{pt,eta,phi,mass}", "Electron.pt"},
    produces={"ht", "n_jet"},
)
def my_producer(self, events, **kwargs): ...

# WRONG — Jet.eta read without being declared
@producer(
    uses={"Jet.pt"},      # missing Jet.eta!
    produces={"ht"},
)
def my_producer(self, events, **kwargs):
    mask = abs(events.Jet.eta) < 2.4   # KeyError at runtime
```

### DO declare every column you write in `produces`

Columns not in `produces` are computed but never saved to disk, and are invisible to downstream tasks.

### DO pass sub-TAF objects in `uses`/`produces` to propagate their columns

```python
# CORRECT — sub_producer's columns are automatically included
@producer(
    uses={sub_producer, "extra_col"},
    produces={sub_producer, "my_output"},
)
def parent(self, events, **kwargs):
    events = self[sub_producer](events, **kwargs)
    ...

# WRONG — sub_producer reads "Jet.pt" but it's not declared here
@producer(
    uses={"extra_col"},
    produces={"my_output"},
)
def parent(self, events, **kwargs):
    events = self[sub_producer](events, **kwargs)  # Jet.pt not loaded!
```

---

## Imports

### DO use `maybe_import` for heavy packages at module level

Columnflow modules are imported in the default (non-columnar) sandbox at setup time. Packages like `awkward`, `numpy`, and `coffea` are not available there.

```python
# CORRECT — deferred import
from columnflow.util import maybe_import
ak = maybe_import("awkward")
np = maybe_import("numpy")

# WRONG — fails at import time in the default sandbox
import awkward as ak
import numpy as np
```

### DO import coffea inside the function body, never at module level

```python
# CORRECT
def my_selector(self, events, **kwargs):
    import coffea.nanoevents.methods.vector
    vec = ak.zip({"pt": events.Jet.pt}, with_name="PtEtaPhiMLorentzVector",
                 behavior=coffea.nanoevents.methods.vector.behavior)

# WRONG
import coffea     # breaks any non-coffea sandbox
```

---

## Selectors

### DO set `results.event` in the exposed Selector

`ReduceEvents` reads `results.event` to determine which events to keep. If it is not set, all events (or no events) may be kept depending on the default.

```python
from operator import and_
from functools import reduce

# After combining all SelectionResult objects:
results.event = reduce(and_, results.steps.values())
```

### DO return `(events, SelectionResult)` from every Selector

```python
# CORRECT
def my_selector(self, events, stats, **kwargs) -> tuple[ak.Array, SelectionResult]:
    ...
    return events, SelectionResult(steps={"jet": mask})

# WRONG — wrong return type breaks ReduceEvents
def my_selector(self, events, stats, **kwargs):
    ...
    return events     # missing SelectionResult
```

### DO use `ak.local_index` to create index arrays for `objects`

```python
# CORRECT — index arrays for SelectionResult.objects
jet_mask = (events.Jet.pt > 25) & (abs(events.Jet.eta) < 2.4)
jet_indices = ak.local_index(events.Jet.pt, axis=1)[jet_mask]

return events, SelectionResult(
    objects={"Jet": {"Jet": jet_indices}},
)

# WRONG — using a boolean mask directly in objects (must be indices)
return events, SelectionResult(
    objects={"Jet": {"Jet": jet_mask}},   # wrong type!
)
```

---

## Configuration

### DO list columns in `cfg.x.keep_columns` that must survive `ReduceEvents`

Any column produced in a Calibrator or Selector that is needed downstream must be explicitly listed. Columns from Producers (run after `ReduceEvents`) are automatically kept.

```python
from columnflow.util import DotDict
from columnflow.columnar_util import ColumnCollection

cfg.x.keep_columns = DotDict.wrap({
    "cf.ReduceEvents": {
        "Jet.{pt,eta,phi,mass,btagDeepFlavB}",
        "Electron.{pt,eta,phi,mass,charge}",
        "Muon.{pt,eta,phi,mass,charge}",
        "MET.{pt,phi}",
        "event", "run", "luminosityBlock",
        ColumnCollection.ALL_FROM_SELECTOR,  # all columns produced in the Selector
    },
})
```

### DO register new files in `law.cfg` under the correct `*_modules` key

After adding a new Python file containing a Selector, Producer, Calibrator, Categorizer, or HistProducer, add it to `law.cfg`. Without this, columnflow cannot find the TAF.

```ini
# After creating myanalysis/selection/jets.py:
selection_modules: ..., myanalysis.selection.{default,jets}

# After creating myanalysis/production/weights.py:
production_modules: ..., myanalysis.production.{default,weights}
```

### DON'T put spaces after commas inside `{}` brace expansions in `law.cfg`

```ini
# CORRECT
selection_modules: myanalysis.selection.{default,jets,trigger}

# WRONG — spaces break the brace expansion
selection_modules: myanalysis.selection.{default, jets, trigger}
```

---

## Systematic Uncertainties

### DO guard MC-only operations with `self.dataset_inst.is_mc`

```python
if self.dataset_inst.is_mc:
    events = self[mc_weight](events, **kwargs)
    events = self[pileup_weight](events, **kwargs)
```

### DON'T hardcode shifted column names — use column aliases

When a shift is active, columnflow swaps column names according to `cfg.x.column_aliases`. Your code should read the nominal name and let the alias mechanism do the substitution.

```python
# WRONG — hardcoded nominal name, broken under mu_up shift
weight = events.muon_weight

# CORRECT — the HistProducer's shift mechanism resolves the alias automatically
# Just declare "muon_weight" in uses and let the framework handle the rest
```

### DO declare shifts in the TAF's `shifts` set (for weight-based uncertainties)

```python
@my_producer.init
def my_producer_init(self: Producer) -> None:
    from columnflow.config_util import get_shifts_from_sources
    self.shifts |= set(get_shifts_from_sources(self.config_inst, "mu"))
```

---

## Versioning and Caching

### DO bump `--version` after changing any code in the pipeline

Law caches outputs based on the version string. If you change a Selector but keep the same version, old cached outputs are reused and your changes have no effect.

```bash
# After changing the Selector:
law run cf.SelectEvents --version v2 --dataset tt_dl_powheg  # not v1!
```

### DO use `--branch 0` for quick local tests before submitting all branches

```bash
# Test on a single file first to catch bugs early
law run cf.SelectEvents --version dev1 --dataset tt_dl_powheg --branch 0

# Then run all files
law run cf.SelectEvents --version dev1 --dataset tt_dl_powheg
```

### DO use `--print-status -1` to understand what will run before running it

```bash
law run cf.PlotVariables1D --version dev1 --processes tt --variables jet1_pt --print-status -1
```

### DO use `--remove-output 0,a,y` to delete and immediately re-run a task

```bash
# Delete CreateHistograms output and re-run it in one command
law run cf.CreateHistograms --version dev1 --variables jet1_pt --remove-output 0,a,y
```

---

## Storage and Columns

### DON'T store large arrays in `cfg.x.*` auxiliaries

Config auxiliaries are for lightweight metadata (thresholds, names, flags). External data (scale factor histograms, efficiency maps) must be loaded at TAF setup time via the `requires`/`setup` hooks.

```python
# WRONG — loading a large array into cfg at config creation time
cfg.x.btag_efficiency = np.load("btag_eff.npy")  # loaded before any task runs

# CORRECT — load it in the setup() hook of the Producer that needs it
@btag_producer.setup
def btag_producer_setup(self, task, reqs, inputs, reader_targets):
    self.btag_eff = inputs["ext_files"]["collection"][0]["btag_eff"].load()
```

### DO produce columns only after `ReduceEvents` where possible

Creating columns in `ProduceColumns` (after reduction) avoids storing them in every file during calibration/selection, saving significant disk space.

### DO use `value_type=np.float32` for large float columns

Float64 uses twice the disk space of float32, with no benefit for typical physics variables.

```python
events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1), value_type=np.float32)
```

---

## Summary Checklist

Before committing or running new analysis code, verify:

- [ ] All columns read from `events` are in `uses`
- [ ] All columns written to `events` are in `produces`
- [ ] `set_ak_column` is used (never `events.field = ...`)
- [ ] `events` is reassigned after every `set_ak_column` and sub-TAF call
- [ ] Heavy imports use `maybe_import`; `coffea` is imported inside function bodies
- [ ] Selectors return `(events, SelectionResult)` with `results.event` set
- [ ] MC-only logic is guarded by `self.dataset_inst.is_mc`
- [ ] All new Python files are registered in `law.cfg` under `*_modules`
- [ ] Columns needed after `ReduceEvents` are in `cfg.x.keep_columns["cf.ReduceEvents"]`
- [ ] No spaces after commas in `law.cfg` brace expansions
- [ ] No Python loops over the `events` array
- [ ] Version bumped after any code change
