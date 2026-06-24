# 05 — Do's and Don'ts for Claude in Columnflow Projects

## Critical Rules

### DO always use `set_ak_column` to modify event data

```python
# CORRECT
events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1))

# WRONG — direct mutation is not supported and may silently fail
events.ht = ak.sum(events.Jet.pt, axis=1)
events["ht"] = ak.sum(events.Jet.pt, axis=1)
```

### DO always declare `uses` and `produces` in TAF decorators

Columnflow uses these sets to track which columns are loaded from disk and which are written. Missing a column in `uses` means it won't be available. Missing it in `produces` means it won't be saved.

```python
# CORRECT — both sets explicitly declared
@producer(
    uses={"Jet.pt", "Jet.eta"},
    produces={"ht", "n_jet"},
)
def my_producer(self, events, **kwargs): ...

# WRONG — omitting uses/produces causes runtime failures
@producer()
def my_producer(self, events, **kwargs): ...
```

### DO use `maybe_import` for heavy packages

```python
# CORRECT
ak = maybe_import("awkward")
np = maybe_import("numpy")

# WRONG — direct import fails when columnflow loads the module in a non-columnar sandbox
import awkward as ak
import numpy as np
```

### DO import coffea inside the function, never at module level

```python
# CORRECT
def my_function(self, events, **kwargs):
    import coffea.nanoevents.methods.vector
    vec = ak.zip({"pt": ...}, with_name="PtEtaPhiMLorentzVector",
                 behavior=coffea.nanoevents.methods.vector.behavior)

# WRONG — coffea at module level breaks non-coffea sandboxes
import coffea
```

### DO register new Python files in `law.cfg`

Every new file containing a Calibrator, Selector, Producer, Categorizer, or HistProducer must be added to the appropriate `*_modules` key in `law.cfg`.

```ini
# CORRECT — after adding myanalysis/selection/jets.py
selection_modules: ..., myanalysis.selection.{default,jets}

# No spaces after commas inside braces!
# WRONG
selection_modules: ..., myanalysis.selection.{default, jets}
```

### DO include sub-TAF's `uses`/`produces` by passing the TAF itself

```python
# CORRECT — propagates all uses/produces from sub_producer
@producer(
    uses={sub_producer, "extra_col"},
    produces={sub_producer, "my_new_col"},
)
def parent(self, events, **kwargs): ...

# WRONG — forgets to re-declare sub_producer columns
@producer(
    uses={"extra_col"},
    produces={"my_new_col"},
)
def parent(self, events, **kwargs):
    events = self[sub_producer](events, **kwargs)  # sub_producer's columns not declared!
```

### DO gate Monte Carlo logic on `self.dataset_inst.is_mc`

```python
# CORRECT
if self.dataset_inst.is_mc:
    events = self[mc_weight](events, **kwargs)

# WRONG — mc_weight column does not exist in data
events = self[mc_weight](events, **kwargs)
```

### DO return `events` from every TAF function

```python
# CORRECT — Producer returns events
def my_producer(self, events, **kwargs) -> ak.Array:
    events = set_ak_column(events, "ht", ...)
    return events

# WRONG — missing return
def my_producer(self, events, **kwargs):
    events = set_ak_column(events, "ht", ...)
```

### DO return `(events, SelectionResult)` from Selectors

```python
# CORRECT
def my_selector(self, events, stats, **kwargs) -> tuple[ak.Array, SelectionResult]:
    ...
    return events, SelectionResult(steps={...})

# WRONG — wrong return type
def my_selector(self, events, stats, **kwargs):
    ...
    return events
```

### DO set `results.event` in the exposed Selector

The `ReduceEvents` task reads `results.event` to filter events. Missing it causes no events to be selected (or an error).

```python
from operator import and_
from functools import reduce

results.event = reduce(and_, results.steps.values())
```

### DO add `keep_columns` in the config for columns created before ReduceEvents

Any column produced by a Calibrator or Selector that should survive `ReduceEvents` must be explicitly listed in `cfg.x.keep_columns` under `"cf.ReduceEvents"`.

```python
cfg.x.keep_columns = DotDict.wrap({
    "cf.ReduceEvents": {
        "Jet.{pt,eta,phi,mass}",
        "my_new_column",                    # REQUIRED if produced in Selector
        ColumnCollection.ALL_FROM_SELECTOR, # convenient catch-all for selector columns
    },
})
```

---

## Architecture Don'ts

### DON'T call `law run` inside Python code

Use the standard `law run cf.<Task>` CLI. If triggering a task programmatically from a script is truly necessary, use the pattern in `best_practices.md` with `task.law_run()`.

### DON'T put analysis logic in `CLAUDE.md` or config files

Analysis code (Selectors, Producers, etc.) belongs in the appropriate subdirectory, not in configuration files or documentation.

### DON'T create a new exposed Selector unless truly needed

One exposed (top-level) Selector should be the entry point, composed of many internal Selectors. Creating multiple exposed Selectors for the same purpose leads to maintenance burden.

### DON'T mix column-level and event-level operations without explicit axis

```python
# WRONG — ambiguous without axis
n_jet = ak.sum(events.Jet.pt > 0)       # sums over EVERYTHING

# CORRECT
n_jet = ak.sum(events.Jet.pt > 0, axis=1)  # per-event count
```

### DON'T use Python loops over events

```python
# WRONG — extremely slow; defeats the columnar paradigm
for event in events:
    ht = sum(jet.pt for jet in event.Jet)

# CORRECT
ht = ak.sum(events.Jet.pt, axis=1)
```

### DON'T modify `events` returned from a sub-TAF call without reassigning

```python
# WRONG — changes are discarded
self[other_producer](events, **kwargs)

# CORRECT
events = self[other_producer](events, **kwargs)
```

---

## Shift / Systematics Don'ts

### DON'T hardcode weight column names in Producers

Use the column alias mechanism so that shifted variants are used automatically when a shift is active:

```python
# WRONG — always uses the nominal weight
weight = events.muon_weight

# CORRECT — alias resolves to muon_weight_up/down under the corresponding shift
weight = events[self.config_inst.x.column_aliases.get("muon_weight", "muon_weight")]
```

Or rely on the `WeightProducer`/`HistProducer` shift mechanism rather than doing manual name resolution.

### DON'T forget to declare shifts in the TAF's `shifts` set

For weight-based and selection-modifying uncertainties, the TAF must declare which shifts it is sensitive to:

```python
@my_producer.init
def my_producer_init(self):
    from columnflow.config_util import get_shifts_from_sources
    self.shifts |= get_shifts_from_sources(self.config_inst, "mu")
```

---

## Config Don'ts

### DON'T add whitespace after commas inside `{}` brace expansions in law.cfg

```ini
# CORRECT
selection_modules: myanalysis.selection.{default,jets,trigger}

# WRONG — spaces break the brace expansion parser
selection_modules: myanalysis.selection.{default, jets, trigger}
```

### DON'T duplicate config entries across Analysis and Config

Put analysis-independent information in the `Campaign` object (year, ecm, tier). Put analysis-specific information in `Config`. Do not copy the same value into both.

### DON'T put large data arrays into Config auxiliaries

`cfg.x.*` is for lightweight metadata (names, thresholds, lookup tables). Heavy data (scale factor histograms, b-tag efficiency arrays) should be loaded from external files at TAF setup time via the `requires`/`setup` hooks.

---

## Style Don'ts

### DON'T import at the function call site what should be a module-level `maybe_import`

```python
# WRONG — hides the deferred import intent
def my_func(self, events, **kwargs):
    import awkward as ak
    ...

# CORRECT — at module level
ak = maybe_import("awkward")
```

### DON'T use `ak.Array` type annotations without the `maybe_import` guard

If `awkward` is not available at import time, type annotations that reference `ak.Array` will fail. Use string annotations or guard them:

```python
# SAFE with maybe_import at module level
ak = maybe_import("awkward")

def my_producer(self: Producer, events: ak.Array, **kwargs) -> ak.Array: ...
```

### DON'T return early from a Producer without returning `events`

```python
# WRONG
if condition:
    return   # forgot to return events!
events = set_ak_column(events, "col", value)
return events

# CORRECT
if condition:
    return events
events = set_ak_column(events, "col", value)
return events
```

---

## Summary Checklist

When writing or reviewing columnflow code, verify:

- [ ] All heavy packages use `maybe_import` at module level
- [ ] `coffea` is imported inside function bodies only
- [ ] Every column read is in `uses`; every column written is in `produces`
- [ ] `set_ak_column` is used (never direct field assignment)
- [ ] `events` is reassigned after every `set_ak_column` or sub-TAF call
- [ ] Selectors return `(events, SelectionResult)` with `results.event` set
- [ ] MC-only logic is guarded by `self.dataset_inst.is_mc`
- [ ] New files are registered in `law.cfg` under the correct `*_modules` key
- [ ] Columns created before `ReduceEvents` are in `cfg.x.keep_columns`
- [ ] No brace-expansion whitespace in `law.cfg`
- [ ] No Python loops over event arrays (use vectorized `ak.*` operations)
