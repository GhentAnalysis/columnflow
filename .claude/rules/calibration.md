---
paths:
  - "**/calibration/**"
---

# Calibrator Rules

## Decorator pattern

```python
from columnflow.calibration import Calibrator, calibrator
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import
ak = maybe_import("awkward")

@calibrator(
    uses={"Jet.{pt,eta,phi,mass,rawFactor,area}", "fixedGridRhoFastjetAll"},
    produces={"Jet.{pt,eta,phi,mass}"},   # nominal; add varied columns in init
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    corrected_pt = events.Jet.pt * self.jec_factor   # loaded in setup()
    events = set_ak_column(events, "Jet.pt", corrected_pt)
    return events
```

## Hooks for external files and shifts

```python
@default.requires
def default_requires(self, task, reqs):
    from columnflow.tasks.external import BundleExternalFiles
    reqs["ext"] = BundleExternalFiles.req(task)

@default.setup
def default_setup(self, task, reqs, inputs, reader_targets):
    bundle = inputs["ext"]["collection"][0]
    # parse JEC file from bundle["jec"] and store correction objects on self
    self.jec_factor = 1.02  # placeholder

@default.init
def default_init(self):
    from columnflow.config_util import get_shifts_from_sources
    # declare which systematic shifts this calibrator is responsible for
    self.shifts |= set(get_shifts_from_sources(self.config_inst, "jec"))
    # add varied output columns for each shift
    for shift_inst in self.shifts:
        self.produces.add(f"Jet.pt_{shift_inst.name}")
```

## Key differences from Producers

- Calibrators run on **raw NanoAOD events** (before selection/reduction).
- They typically **overwrite** kinematic columns (e.g. `Jet.pt`) rather than creating new named columns.
- Separate task branches run for each registered systematic shift.
- Varied columns (e.g. `Jet.pt_jec_up`) must be declared in `produces` dynamically in the `init` hook.
- Column aliases in `cfg` map `"Jet.pt"` → `"Jet.pt_jec_up"` when the `jec_up` shift is active.
