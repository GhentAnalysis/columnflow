# 03 — Task Commands

All tasks are invoked with `law run cf.<TaskName>`. The `--version` flag is always required. `--config` and `--analysis` can be set in `law.cfg` as defaults.

Use `--help` on any task for the full parameter list:
```bash
law run cf.SelectEvents --help
```

---

## Universal Flags

| Flag | Purpose |
|---|---|
| `--version <name>` | Version tag for output paths (required) |
| `--config <name>` | Config name (default: `law.cfg default_config`) |
| `--analysis <name>` | Analysis module path (default: `law.cfg default_analysis`) |
| `--branch <int>` | Run a single branch (file) instead of all |
| `--print-status -1` | Show task tree + output existence (recursive) |
| `--print-output <depth>` | Show output file paths at given depth |
| `--remove-output <depth>,<mode>[,y]` | Remove outputs; modes: `a` all, `i` interactive, `d` dry-run |
| `--workflow htcondor` | Submit to HTCondor |
| `--workflow slurm` | Submit to Slurm |
| `--pilot` | Run one branch to check before launching all |

---

## Step-by-step Commands

### GetDatasetLFNs

Resolves logical file names for a dataset (requires GRID proxy for CMS data).

```bash
law run cf.GetDatasetLFNs \
    --version dev1 \
    --dataset tt_dl_powheg
```

---

### CalibrateEvents

Applies calibrations (e.g. jet energy corrections). Runs per file/branch.

```bash
law run cf.CalibrateEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --calibrators default \
    --shift nominal

# Single branch test
law run cf.CalibrateEvents --version dev1 --dataset tt_dl_powheg --branch 0

# With a systematic shift
law run cf.CalibrateEvents --version dev1 --dataset tt_dl_powheg --shift jec_up
```

---

### SelectEvents

Runs the Selector; produces event/object masks (parquet) and selection statistics (json).

```bash
law run cf.SelectEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --selector default \
    --calibrators default

# Verify outputs recursively
law run cf.SelectEvents --version dev1 --dataset tt_dl_powheg --print-status -1

# Cutflow plot requires this first (SelectEvents at cutflow step)
law run cf.SelectEvents --version dev1 --config l18
```

---

### ReduceEvents

Applies selection masks to columns; writes the reduced parquet files.

```bash
law run cf.ReduceEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --selector default \
    --calibrators default

# Single branch (first file) — quick local test
law run cf.ReduceEvents --version dev1 --branch 0
```

---

### MergeReducedEvents

Merges per-file reduced parquet files into per-dataset merged files.

```bash
law run cf.MergeReducedEvents \
    --version dev1 \
    --dataset tt_dl_powheg
```

---

### ProduceColumns

Runs Producer(s) on the merged reduced events; writes additional column parquet files.

```bash
law run cf.ProduceColumns \
    --version dev1 \
    --dataset tt_dl_powheg \
    --producers default

# Multiple producers (comma-separated, order matters for column overwriting)
law run cf.ProduceColumns \
    --version dev1 \
    --dataset tt_dl_powheg \
    --producers default,extra_features
```

---

### CreateHistograms

Fills histograms for the given variables, categories, and shift.

```bash
law run cf.CreateHistograms \
    --version dev1 \
    --dataset tt_dl_powheg \
    --variables jet1_pt,n_jet \
    --categories incl \
    --shift nominal \
    --producers default \
    --hist-producer default

# All shifts at once (nominal + all registered shifts)
law run cf.CreateHistograms \
    --version dev1 \
    --datasets tt_dl_powheg,wjets_madgraph \
    --variables jet1_pt \
    --shifts nominal,jec_up,jec_down,mu_up,mu_down
```

---

### MergeHistograms

Merges per-dataset histogram files.

```bash
law run cf.MergeHistograms \
    --version dev1 \
    --variables jet1_pt \
    --processes tt,wjets
```

---

### PlotVariables1D

Creates 1D variable plots per process, stacked or overlaid.

```bash
law run cf.PlotVariables1D \
    --version dev1 \
    --processes tt,wjets,data \
    --variables jet1_pt,n_jet \
    --categories incl \
    --shift nominal

# Save to PDF and PNG
law run cf.PlotVariables1D \
    --version dev1 \
    --processes tt \
    --variables jet1_pt \
    --file-types pdf,png

# With custom plot function
law run cf.PlotVariables1D \
    --version dev1 \
    --processes tt \
    --variables jet1_pt \
    --plot-function myanalysis.plotting.my_plot_func

# Open plots automatically
law run cf.PlotVariables1D \
    --version dev1 \
    --processes tt \
    --variables jet1_pt \
    --view-cmd evince
```

---

### PlotVariables2D

Creates 2D plots.

```bash
law run cf.PlotVariables2D \
    --version dev1 \
    --processes tt \
    --variables jet1_pt__jet1_eta \
    --categories incl
```

---

### PlotShiftedVariables1D

Shows nominal + up/down shift variations for a variable.

```bash
law run cf.PlotShiftedVariables1D \
    --version dev1 \
    --processes tt \
    --variables jet1_pt \
    --shift-sources jec,mu \
    --categories incl
```

---

### PlotCutflow

Plots total event yield at each selection step.

```bash
law run cf.PlotCutflow \
    --version dev1 \
    --datasets tt_dl_powheg \
    --selector-steps muon,jet,bjet
```

---

### PlotCutflowVariables1D

Plots a variable distribution at each selector step.

```bash
law run cf.PlotCutflowVariables1D \
    --version dev1 \
    --datasets tt_dl_powheg \
    --processes tt \
    --variables genTop_pt \
    --categories incl \
    --skip-ratio
```

---

### CreateDatacards

Produces CMS combine-compatible datacards and ROOT shape files.

```bash
law run cf.CreateDatacards \
    --version dev1 \
    --inference-model example \
    --variables jet1_pt \
    --categories incl
```

---

## Useful Utility Commands

```bash
# Check if a task's output already exists
law run cf.SelectEvents --version dev1 --dataset tt_dl_powheg --print-status 0

# Remove just the histogram task output and re-run it
law run cf.CreateHistograms --version dev1 --variables jet1_pt --remove-output 0,a,y

# List all available law tasks
law index --verbose

# Get help on a specific task
law run cf.ProduceColumns --help

# Run inside the columnar sandbox manually (for debugging)
cf_sandbox venv_columnar_dev bash
```

---

## Remote Execution (HTCondor)

```bash
# Submit SelectEvents for all files of a dataset to HTCondor
law run cf.SelectEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --workflow htcondor

# Monitor running jobs
law run cf.SelectEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --workflow htcondor \
    --print-status -1
```

---

## Pinning Upstream Task Versions

Use task-family-specific version flags to pin upstream outputs:

```bash
# Use v1 outputs from CalibrateEvents while running SelectEvents as v2
law run cf.SelectEvents \
    --version v2 \
    --cf.CalibrateEvents-version v1 \
    --dataset tt_dl_powheg
```

---

## Dataset / Process Wildcards

Many tasks accept glob patterns:

```bash
# All tt datasets
law run cf.PlotVariables1D --version dev1 --datasets "tt*" --variables jet1_pt

# Multiple processes
law run cf.PlotVariables1D --version dev1 --processes "tt,wjets,dy*" --variables n_jet
```

---

## Full Pipeline — Single Command Chain

Running the plotting task automatically triggers all upstream tasks:

```bash
# This single command triggers the full pipeline for one dataset + one variable
law run cf.PlotVariables1D \
    --version dev1 \
    --datasets tt_dl_powheg \
    --variables jet1_pt \
    --processes tt \
    --categories incl \
    --shift nominal \
    --branch 0   # single branch for a quick test
```
