---
paths:
  - "**/histogramming/**"
---

# HistProducer Rules

## Extending cf_default (standard approach)

```python
from columnflow.histogramming import HistProducer
from columnflow.histogramming.default import cf_default
from columnflow.columnar_util import Route
from columnflow.util import maybe_import
from columnflow.config_util import get_shifts_from_sources

ak = maybe_import("awkward")
np = maybe_import("numpy")


@cf_default.hist_producer()
def default(self: HistProducer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """Return (events, per-event weight array)."""
    weight = ak.Array(np.ones(len(events), dtype=np.float32))

    if self.dataset_inst.is_mc and len(events):
        for column in self.weight_columns:
            weight = weight * Route(column).apply(events)

    return events, weight


@default.init
def default_init(self: HistProducer) -> None:
    self.weight_columns = set()
    if self.dataset_inst.is_data:
        return

    # Add weight column names; declared in uses so they are loaded from disk
    self.weight_columns |= {"normalization_weight", "muon_weight", "pu_weight"}
    self.uses |= self.weight_columns

    # Declare which shift sources affect the event weight (for alias resolution)
    self.shifts |= set(get_shifts_from_sources(self.config_inst, "mu", "minbias_xs"))
```

## Key rules

- The `__call__` method returns `(events, weight_array)` — not just events.
- The weight is a 1D float32 array of length `len(events)`. Data events get weight = 1.
- Add weight columns to `self.uses` (not the decorator `uses`) in the `init` hook so they are dynamically included.
- Declare `self.shifts` in the `init` hook: the framework uses this to resolve column aliases when a shift is active (e.g. `muon_weight` → `muon_weight_up` under `mu_up`).
- `Route(column).apply(events)` safely reads a column; use it instead of `events[column]` when the column name is a variable.
- Register in `law.cfg` under `hist_production_modules`.
