---
paths:
  - "**/config/**"
---

# Config Object Rules

## Hierarchy: Analysis → Config → Campaign → Dataset

```python
import order as od

analysis = od.Analysis(name="my_analysis", id=1)
cpn = od.Campaign(name="run2_2018", id=4, ecm=13, aux={"tier": "NanoAOD", "year": 2018})
cfg = analysis.add_config(cpn, name="run2_2018", id=4)
```

## Adding order objects to Config

```python
# Process (must be added before datasets that reference it)
cfg.add_process(procs.tt)

# Dataset (fetched from Campaign by name)
cfg.add_dataset(cpn.get_dataset("tt_dl_powheg"))

# Shift
cfg.add_shift(name="nominal", id=0)
cfg.add_shift(name="mu_up", id=1, type=od.Shift.SHAPE)

# Variable
cfg.add_variable(
    name="jet1_pt",
    expression="Jet.pt[:,0]",   # awkward expression evaluated in HistProducer
    null_value=EMPTY_FLOAT,
    binning=(40, 0.0, 400.0),
    unit="GeV",
    x_title=r"Leading jet $p_T$",
)

# Category
from columnflow.config_util import add_category
add_category(cfg, name="incl", id=1, selection="cat_incl", label="Inclusive")
```

## Required auxiliaries (cfg.x.*)

```python
from scinum import Number
cfg.x.luminosity = Number(59740, {"lumi_13TeV_2018": 0.025j})

from columnflow.util import DotDict
from columnflow.columnar_util import ColumnCollection

cfg.x.keep_columns = DotDict.wrap({
    "cf.ReduceEvents": {
        "Jet.{pt,eta,phi,mass,btagDeepFlavB}",
        "Electron.{pt,eta,phi,mass,charge}",
        "Muon.{pt,eta,phi,mass,charge}",
        "MET.{pt,phi}",
        "event", "run", "luminosityBlock",
        ColumnCollection.ALL_FROM_SELECTOR,
    },
})

# Default CLI argument values (override on command line)
cfg.x.default_calibrator = "default"
cfg.x.default_selector   = "default"
cfg.x.default_producer   = "default"
cfg.x.default_variables  = ("n_jet", "jet1_pt")
```

## Shift aliases (weight-based uncertainties)

```python
from columnflow.config_util import add_shift_aliases

# Maps "muon_weight" → "muon_weight_up" when mu_up shift is active
add_shift_aliases(cfg, "mu", {
    "muon_weight": "muon_weight_{direction}",
})
```

## External files

```python
cfg.x.external_files = DotDict.wrap({
    "muon_sf": "/path/to/muon_sf.json",
    "btag_sf": ("/path/to/btag_sf.json", "v2"),   # with version
})
```

## Variable expression formats

```python
"Jet.pt[:,0]"    # leading jet pT (EMPTY_FLOAT if no jets — set null_value)
"Jet.pt"         # all jets flattened (1D histogram across all events and jets)
"n_jet"          # scalar column (from ProduceColumns)
"MET.pt"         # scalar per-event field
```
