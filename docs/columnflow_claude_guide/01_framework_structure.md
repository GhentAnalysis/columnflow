# 01 — Columnflow Framework: The Big Picture

## What is columnflow?

Columnflow is a Python framework for **fully orchestrated, columnar High Energy Physics analyses**. It sits between raw NanoAOD ROOT files and final statistical results (plots, datacards), handling the entire processing chain automatically.

It solves three practical problems every HEP analyst faces:

| Problem | How columnflow solves it |
|---|---|
| **Scalability** — datasets contain billions of events that do not fit in memory | Processes events in chunks; submits to HTCondor/Slurm automatically |
| **Reproducibility** — analysts need to re-run with different settings and track intermediate results | Every task output is versioned and cached; re-running only re-processes what changed |
| **Systematic uncertainties** — dozens of weight variations and kinematic corrections must be propagated consistently | First-class support for shifts; the task graph branches automatically per systematic |

---

## The Three Pillars

### 1. Law — Workflow orchestration

[Law](https://github.com/riga/law) (built on [Luigi](https://luigi.readthedocs.io)) defines the **task graph**: every processing step is a `Task` object with declared outputs. When you run a downstream task (e.g. `PlotVariables1D`), law checks which upstream tasks are missing and runs them first — automatically.

Key properties:
- Tasks are **idempotent**: if the output already exists, the task is skipped.
- Tasks run **locally** or on **HTCondor / Slurm** with a single flag change (`--workflow htcondor`).
- The `--version` tag separates parallel analysis branches on disk — you can have `dev1` and `prod1` coexist.

### 2. Order — Metadata management

[Order](https://github.com/riga/order) provides Python objects for HEP bookkeeping: `Analysis`, `Campaign`, `Config`, `Dataset`, `Process`, `Variable`, `Shift`, `Category`. These objects replace scattered config files with a typed, queryable in-memory structure. Every analysis object (process colour, cross-section, dataset file count) lives here.

### 3. Awkward-array — Columnar event data

All event data is stored as [awkward arrays](https://awkward-array.org): irregular (jagged) arrays where each event can have a different number of jets, leptons, etc. Operations are vectorized — you never write a Python loop over events. The `events` array is the central object passed through the entire pipeline.

---

## The Analysis Pipeline

The standard pipeline processes data from ROOT files to plots and datacards:

```
Raw NanoAOD ROOT files
         │
         ▼
GetDatasetLFNs          Resolves file paths (LFNs) from DAS or custom source
         │
         ▼
CalibrateEvents         Applies corrections (JEC, MET corrections, tau ES, ...)
         │
         ▼
SelectEvents            Defines event and object masks; saves selection statistics
         │
         ▼
ReduceEvents            Applies masks; writes reduced Parquet files (~100× smaller)
         │
         ▼
MergeReducedEvents      Merges per-file outputs into one file per dataset
         │
         ├──────────────────────────────────────────────────┐
         ▼                                                  ▼
ProduceColumns                                    CreateHistograms (directly)
(creates extra columns)
         │
         ▼
CreateHistograms        Fills Hist histograms for all variables / categories / shifts
         │
         ▼
MergeHistograms         Merges histograms across branches (files) of a dataset
         │
         ├────────────────────────┐
         ▼                        ▼
  PlotVariables1D         CreateDatacards
  (matplotlib plots)      (CMS combine format)
```

### What happens at each step

**GetDatasetLFNs** — Queries CMS DAS (or a custom function) to find the ROOT file paths for each dataset. Saves them as a JSON file used by all downstream tasks. Requires a valid GRID proxy for CMS data.

**CalibrateEvents** — Runs user-defined `Calibrator` objects on raw events. Calibrators add corrected columns (e.g. `Jet.pt` after JEC) without removing the originals. Multiple calibrators can run sequentially. For systematic variations that modify kinematics (e.g. JEC up/down), separate task branches are created.

**SelectEvents** — Runs one `Selector` on the calibrated events. The selector produces:
- Boolean event masks (one per selection step, e.g. `"trigger"`, `"muon"`, `"jet"`)
- Index arrays for object collections (e.g. which jets pass `pt > 25 GeV && |eta| < 2.4`)
- A `stats.json` file with event counts and MC weight sums (needed for normalization later)

Masks are saved to Parquet — they are **not yet applied** here.

**ReduceEvents** — Applies the masks from `SelectEvents` to all columns. Writes a compact Parquet file containing only selected events and selected objects. Columns not listed in `cfg.x.keep_columns["cf.ReduceEvents"]` are dropped here permanently.

**MergeReducedEvents** — Merges the many small per-file Parquet outputs into larger files per dataset, targeting a configurable size (default ~512 MB, set via `cfg.x.reduced_file_size`).

**ProduceColumns** — Runs `Producer` objects on the merged reduced events to create new high-level columns: `ht`, `n_bjet`, `mll`, category IDs, event weights, ML scores, etc. These are saved in separate Parquet files that are transparently merged with the reduced events by downstream tasks.

**CreateHistograms** — Fills `Hist` histograms for all requested variables, categories, and systematic shifts. The `HistProducer` controls event weighting and histogram filling logic. Each histogram is labelled by dataset, shift, and category.

**MergeHistograms** — Merges histograms from all branches (files) of a dataset. A second merge step (`MergeShiftedHistograms`) further combines nominal and shifted histograms across all datasets for a given process, ready for plotting or inference.

**PlotVariables1D / 2D** — Creates matplotlib/mplhep-styled plots from merged histograms. Supports stacked MC + data overlays, ratio panels, and shifted variable comparisons.

**CreateDatacards** — Produces CMS `combine`-compatible datacards (`.txt`) and shape ROOT files from the merged histograms, driven by an `InferenceModel` object.

---

## Five Task Array Function (TAF) Types

User-defined code hooks into the pipeline through **Task Array Functions**. Each TAF type slots into one specific task:

| TAF | Class | Task | CLI parameter | Quantity |
|---|---|---|---|---|
| Calibrator | `Calibrator` | `CalibrateEvents` | `--calibrators` | 0 or more |
| Selector | `Selector` | `SelectEvents` | `--selector` | exactly 1 |
| Reducer | `Reducer` | `ReduceEvents` | `--reducer` | exactly 1 |
| Producer | `Producer` | `ProduceColumns` | `--producers` | 0 or more |
| HistProducer | `HistProducer` | `CreateHistograms` | `--hist-producer` | exactly 1 |

All TAFs share the same decorator pattern and lifecycle. See [02 Coding Style](02_coding_style.md) for the full pattern.

---

## Directory Structure of an Analysis

```
myanalysis/
├── analysis/
│   └── my_analysis.py        # Creates Analysis, Campaign, Config objects
├── config/
│   ├── processes.py          # Process definitions (name, xsec, colour)
│   ├── datasets.py           # Dataset definitions (files, keys)
│   ├── variables.py          # Variable definitions (binning, expression)
│   └── categories.py        # Category and Categorizer definitions
├── calibration/
│   └── jets.py               # Custom Calibrator(s)
├── selection/
│   ├── default.py            # Exposed (top-level) Selector
│   ├── objects.py            # Internal object-selection Selectors
│   └── stats.py              # Stats-increment helper
├── production/
│   ├── default.py            # Main exposed Producer (called from CLI)
│   ├── weights.py            # Event weight Producers
│   └── features.py           # High-level variable Producers
├── categorization/
│   └── categories.py         # Categorizer definitions
├── histogramming/
│   └── default.py            # HistProducer
├── inference/
│   └── default.py            # InferenceModel for datacards
├── tasks/                    # Custom analysis-specific law tasks
├── law.cfg                   # Workflow configuration (must register all modules)
└── setup.sh                  # Environment setup
```

---

## Configuration Objects (order)

### Hierarchy

```
Analysis  ─── top-level container (rarely holds analysis logic itself)
  └── Config  ─── analysis + campaign combination; carries all per-config settings
        └── Campaign  ─── experimental period (year, energy, tier, file locations)
              └── Dataset  ─── one Monte Carlo or data sample
```

### Analysis

```python
import order as od
analysis = od.Analysis(name="hbw", id=1)
```

### Campaign

```python
cpn = od.Campaign(
    name="run2_2018",
    id=4,
    ecm=13,
    aux={
        "tier": "NanoAOD",
        "year": 2018,
        "location": "root://xrootd-cms.infn.it//",
    },
)
```

### Process

```python
from scinum import Number

tt = od.Process(
    name="tt",
    id=1000,
    label=r"$t\bar{t}$",
    color=(205, 0, 9),
    xsecs={13: Number(831.76, {"scale": (19.77, 29.20), "pdf": 35.06})},
)
```

### Dataset

```python
cpn.add_dataset(
    name="tt_dl_powheg",
    id=1,
    processes=[procs.tt],
    info={
        "nominal": od.DatasetInfo(
            keys=["/TTTo2L2Nu.../RunIISummer20UL18NanoAODv9.../NANOAODSIM"],
            n_files=242,
            n_events=276079127,
        ),
        # For dedicated-dataset systematics, add extra info entries:
        "tune_up": od.DatasetInfo(
            keys=["/TTTo2L2Nu_TuneUp.../NANOAODSIM"],
            n_files=30,
            n_events=10000000,
        ),
    },
)
```

### Config

The `Config` object links an Analysis and a Campaign and carries all per-config settings:

```python
cfg = analysis.add_config(campaign, name="run2_2018", id=4)

# --- Required order objects ---
cfg.add_process(procs.tt)
cfg.add_dataset(campaign.get_dataset("tt_dl_powheg"))
cfg.add_shift(name="nominal", id=0)

# Variable definition
cfg.add_variable(
    name="jet1_pt",
    expression="Jet.pt[:,0]",
    null_value=EMPTY_FLOAT,
    binning=(40, 0.0, 400.0),
    unit="GeV",
    x_title=r"Leading jet $p_T$",
)

# Category
from columnflow.config_util import add_category
add_category(cfg, name="incl", id=1, selection="cat_incl", label="Inclusive")

# --- Auxiliary configuration (cfg.x.*) ---
from columnflow.util import DotDict
from columnflow.columnar_util import ColumnCollection

cfg.x.luminosity = Number(59740, {"lumi_13TeV_2018": 0.025j})

cfg.x.keep_columns = DotDict.wrap({
    "cf.ReduceEvents": {
        "{Jet,FatJet}.{pt,eta,phi,mass,btagDeepFlavB}",
        "Electron.{pt,eta,phi,mass,charge}",
        "Muon.{pt,eta,phi,mass,charge}",
        "MET.{pt,phi}",
        "event", "run", "luminosityBlock",
        ColumnCollection.ALL_FROM_SELECTOR,
    },
})

cfg.x.default_calibrator = "default"
cfg.x.default_selector = "default"
cfg.x.default_producer = "default"
cfg.x.default_variables = ("n_jet", "jet1_pt")
```

### Important Config Auxiliaries

| Key | Type | Purpose |
|---|---|---|
| `cfg.x.keep_columns` | `DotDict` of sets | Columns that survive `ReduceEvents` |
| `cfg.x.luminosity` | `scinum.Number` | Luminosity with uncertainties |
| `cfg.x.external_files` | `DotDict` | Paths/URLs to external scale-factor files |
| `cfg.x.get_dataset_lfns` | callable | Custom LFN-retrieval function |
| `cfg.x.default_calibrator` | str | Default `--calibrators` value |
| `cfg.x.default_selector` | str | Default `--selector` value |
| `cfg.x.default_producer` | str/tuple | Default `--producers` value |
| `cfg.x.default_variables` | tuple | Default `--variables` value |
| `cfg.x.reduced_file_size` | float | Target merged-file size in MB |
| `cfg.x.versions` | dict | Pinned task versions (see best_practices) |

---

## Systematic Uncertainties (Shifts)

Columnflow has first-class support for three classes of systematics:

### Rate-only uncertainties

Affect the overall yield only (e.g. luminosity uncertainty). Defined entirely in the inference model — no extra task branches needed.

```python
# In the inference model:
model.add_parameter("lumi", type="lnN", effect=1.025)
```

### Weight-based uncertainties (shape)

Applied as event weight variations (e.g. muon scale factors, pile-up weights). They produce separate histograms for up/down variations without re-running the selection.

```python
# In config:
cfg.add_shift(name="mu_up", id=1, type=od.Shift.SHAPE)
cfg.add_shift(name="mu_down", id=2, type=od.Shift.SHAPE)

from columnflow.config_util import add_shift_aliases
add_shift_aliases(cfg, "mu", {"muon_weight": "muon_weight_{direction}"})

# In the weight Producer's init hook:
from columnflow.config_util import get_shifts_from_sources
self.shifts |= get_shifts_from_sources(self.config_inst, "mu")
```

### Selection-modifying uncertainties (e.g. JEC)

Kinematic corrections that affect object selection. A separate task branch runs the complete pipeline (from `CalibrateEvents` onwards) for each variation.

```python
# In config:
cfg.add_shift(name="jec_up", id=10, type=od.Shift.SHAPE, tags={"selection_dependent"})
cfg.add_shift(name="jec_down", id=11, type=od.Shift.SHAPE, tags={"selection_dependent"})

# Column aliases map "Jet.pt" → "Jet.pt_jec_up" when jec_up shift is active:
add_shift_aliases(cfg, "jec", {"Jet.pt": "Jet.pt_{name}", "Jet.eta": "Jet.eta_{name}"})

# In the Selector's init hook:
from columnflow.config_util import get_shifts_from_sources
self.shifts |= get_shifts_from_sources(self.config_inst, "jec", tags={"selection_dependent"})
```

### Dedicated-dataset uncertainties

A completely separate Monte Carlo sample is processed (e.g. tune, hdamp variations). The dataset `info` dict key must match the shift name exactly.

```python
cfg.add_shift(name="tune_up", id=20, type=od.Shift.SHAPE, tags={"disjoint_from_nominal"})
# The dataset info key "tune_up" automatically links this shift to the correct dataset.
```

---

## Data Flow Summary

```
ROOT files  →  chunks of ak.Array named "events"
                     │
                [Calibrator]   adds corrected columns (never removes)
                     │
                [Selector]     creates boolean masks → stats.json + masks.parquet
                     │
                [Reducer]      applies masks → reduced_events.parquet
                     │
                [Producer]     adds columns → extra_columns.parquet
                     │
                [HistProducer] event weights + fills Hist → histograms.pkl
                     │
                [Plots / Datacards]
```

All intermediate data is stored in a directory tree under the path configured in `law.cfg`, organised by task family, version, config, dataset, and shift.
