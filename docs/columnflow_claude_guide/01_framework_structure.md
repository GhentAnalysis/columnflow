# 01 — Columnflow Framework Structure

## Overview

Columnflow is a backend for columnar, fully orchestrated HEP analyses in pure Python.

| Layer | Package | Role |
|---|---|---|
| Workflow orchestration | [law](https://github.com/riga/law) | Task graph, CLI, remote execution |
| Metadata / bookkeeping | [order](https://github.com/riga/order) | Analysis, Campaign, Config, Dataset, Process, Variable, Shift |
| Columnar data | [awkward-array](https://awkward-array.org) | Event arrays; all physics data |
| Histogramming | [Hist](https://hist.readthedocs.io) | Histogram objects produced by `CreateHistograms` |
| Physics objects | [coffea](https://coffeateam.github.io/coffea/) | Lorentz vector behaviour on awkward arrays |

---

## Analysis Pipeline (Task Graph)

The standard linear pipeline from raw data to plots:

```
GetDatasetLFNs
     │
CalibrateEvents          ← applies calibrations (e.g. JEC)
     │
SelectEvents             ← creates event and object masks; produces stats.json
     │
ReduceEvents             ← applies masks; writes reduced parquet files
     │
MergeReducedEvents       ← merges per-file reduced files into one file per dataset
     │
ProduceColumns           ← creates additional high-level columns
     │
CreateHistograms         ← fills Hist histograms per variable / category / shift
     │
MergeHistograms          ← merges per-dataset histograms
     │
MergeShiftedHistograms   ← merges nominal + shifted histograms for inference
     │
PlotVariables1D          ← produces matplotlib/mplhep plots
CreateDatacards          ← produces CMS combine datacards + ROOT shape files
```

Tasks with `cf.` prefix are columnflow built-ins; user-defined tasks live in `<analysis>/tasks/`.

### Key properties

- **Chunking**: events are processed in chunks of ≤100 000 events (set in `law.cfg` under `chunked_io_chunk_size`).
- **Parallelism**: each file / chunk is a separate law task branch; submit to HTCondor/Slurm with `--workflow htcondor`.
- **Reproducibility**: `--version` tags all intermediate outputs; different versions coexist on disk.
- **Columnar storage**: intermediate results are stored as **Parquet** files (events) or **pickle** files (histograms).

---

## Five Task Array Function (TAF) Types

| TAF type | Class | CLI parameter | Quantity | Task |
|---|---|---|---|---|
| Calibrator | `Calibrator` | `--calibrators` | 0..N | `CalibrateEvents` |
| Selector | `Selector` | `--selector` | exactly 1 | `SelectEvents` |
| Reducer | `Reducer` | `--reducer` | exactly 1 | `ReduceEvents` |
| Producer | `Producer` | `--producers` | 0..N | `ProduceColumns` |
| HistProducer | `HistProducer` | `--hist-producer` | exactly 1 | `CreateHistograms` |

All TAFs share the same decorator pattern — see [02 Coding Style](02_coding_style.md).

---

## Configuration Objects (order package)

### Hierarchy

```
Analysis
  └── Config (links Analysis + Campaign)
        └── Campaign
              └── Dataset → DatasetInfo (files, events, keys)
```

### Analysis

Top-level container; rarely holds analysis logic itself.

```python
import order as od
analysis = od.Analysis(name="my_analysis", id=1)
```

### Campaign

Experiment-period-specific information (year, energy, tier, file locations).

```python
cpn = od.Campaign(
    name="run3_2022",
    id=1,
    ecm=13.6,
    aux={
        "tier": "NanoAOD",
        "year": 2022,
        "location": "root://...",
    },
)
```

### Config

Analysis + campaign combination; carries all per-config parameters.

```python
cfg = analysis.add_config(campaign, name="run3_2022", id=1)
```

### Process

Physical process with cross-section and plot metadata.

```python
from scinum import Number
proc = od.Process(
    name="tt",
    id=1000,
    label=r"$t\bar{t}$",
    color=(128, 76, 153),
    xsecs={13.6: Number(831.76, {"scale": (19.77, 29.20)})},
)
```

### Dataset

A sample linked to a Campaign and one or more Processes.

```python
cpn.add_dataset(
    name="tt_dl_powheg",
    id=1,
    processes=[procs.tt],
    info={
        "nominal": od.DatasetInfo(
            keys=["/TT.../NANOAODSIM"],
            n_files=242,
            n_events=276079127,
        ),
    },
)
```

### Variable

Defines the column expression and histogram binning for `CreateHistograms`.

```python
cfg.add_variable(
    name="jet1_pt",
    expression="Jet.pt[:,0]",
    null_value=EMPTY_FLOAT,
    binning=(40, 0.0, 400.0),
    unit="GeV",
    x_title=r"Jet $p_{T}$",
)
```

### Shift

Systematic uncertainty variant (rate-only, weight-based, or dedicated dataset).

```python
cfg.add_shift(name="nominal", id=0)
cfg.add_shift(name="mu_up", id=1, type=od.Shift.SHAPE)
cfg.add_shift(name="mu_down", id=2, type=od.Shift.SHAPE)
```

### Category / Categorizer

Analysis phase-space regions used in histogramming and plotting.

```python
from columnflow.config_util import add_category
add_category(cfg, name="incl", id=1, selection="cat_incl", label="Inclusive")
```

---

## Important Config Auxiliaries

These `cfg.x.*` entries have special meaning in columnflow:

| Key | Type | Purpose |
|---|---|---|
| `cfg.x.keep_columns` | `DotDict` of sets | Which columns survive `ReduceEvents` |
| `cfg.x.luminosity` | `scinum.Number` | Luminosity with uncertainties |
| `cfg.x.external_files` | `DotDict` | Paths/URLs to external scale-factor files |
| `cfg.x.get_dataset_lfns` | callable | Custom LFN-retrieval function |
| `cfg.x.default_calibrator` | str | Default `--calibrator` value |
| `cfg.x.default_selector` | str | Default `--selector` value |
| `cfg.x.default_producer` | str/tuple | Default `--producer` value |
| `cfg.x.default_variables` | tuple | Default `--variables` value |
| `cfg.x.reduced_file_size` | float | Target merged-file size in MB |
| `cfg.x.versions` | dict | Pinned task versions |

### keep_columns example

```python
from columnflow.util import DotDict
from columnflow.columnar_util import ColumnCollection

cfg.x.keep_columns = DotDict.wrap({
    "cf.ReduceEvents": {
        "{Jet,FatJet}.{pt,eta,phi,mass,btagDeepFlavB}",
        "Electron.{pt,eta,phi,mass,charge}",
        "Muon.{pt,eta,phi,mass,charge}",
        "MET.{pt,phi}",
        "event", "run", "luminosityBlock",
        ColumnCollection.ALL_FROM_SELECTOR,
    },
    "cf.ProduceColumns": {
        "ht", "n_jet",
    },
})
```

---

## law.cfg Structure

The `law.cfg` file drives the workflow. Critical sections:

```ini
[analysis]
default_analysis: myanalysis.analysis.my_analysis.my_analysis
default_config: run3_2022
default_dataset: tt_dl_powheg

# Register all Python modules columnflow should know about
calibration_modules:   columnflow.calibration.cms.{jets,met}, myanalysis.calibration.default
selection_modules:     columnflow.selection.empty, columnflow.selection.cms.{json_filter,met_filters}, myanalysis.selection.default
reduction_modules:     columnflow.reduction.default
production_modules:    columnflow.production.{categories,normalization,processes}, myanalysis.production.default
categorization_modules: myanalysis.categorization.example
hist_production_modules: columnflow.histogramming.default, myanalysis.histogramming.example
inference_modules:     columnflow.inference, myanalysis.inference.example

[outputs]
# Map tasks to storage locations
cf.CalibrateEvents: local, /path/to/store
cf.SelectEvents:    local, /path/to/store
cf.ReduceEvents:    local, /path/to/store
cf.ProduceColumns:  local, /path/to/store
cf.CreateHistograms: local, /path/to/store
```

**Critical**: after adding any new Python file (calibrator, selector, producer, etc.) you **must** register it in `law.cfg` under the correct `*_modules` key. No spaces after commas inside `{}` brace expansions.

---

## Systematic Uncertainties (Shifts)

Three classes, ordered by complexity:

### 1. Rate-only uncertainties
Only affect yields, not selection. Defined entirely in the inference model. No additional workflow steps needed.

### 2. Weight-based (shape) uncertainties
Applied via event weights. Require:
1. `cfg.add_shift(...)` for up/down variants
2. `add_shift_aliases(cfg, "source_name", ...)` to map column names
3. The `WeightProducer` / `HistProducer` must declare the shift in its `shifts` set
4. Considered only from `CreateHistograms` onwards

### 3. Selection-modifying uncertainties (e.g. JEC)
Propagate through the entire pipeline. Require:
1. `cfg.add_shift(...)` for up/down variants with `tags={"selection_dependent"}`
2. Column aliases for varied kinematic columns
3. The `Selector` (and upstream `Calibrator`) must declare the shifts in their `shifts` set
4. Separate task branches are run for each shift

### 4. Dedicated-dataset uncertainties (e.g. tune, hdamp)
A completely separate dataset is processed for each variation. The dataset `info` dict key must match the shift name.
