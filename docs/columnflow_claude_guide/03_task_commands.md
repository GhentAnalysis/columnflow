# 03 — User Guide: Working with Columnflow Tasks

All tasks are run with `law run cf.<TaskName>`. Law automatically determines and runs any missing upstream tasks before the requested one.

## Universal Flags

| Flag | Purpose | Notes |
|---|---|---|
| `--version <name>` | Version tag for output paths | Required for every task |
| `--config <name>` | Which config to use | Default set in `law.cfg` |
| `--analysis <module>` | Analysis module path | Default set in `law.cfg` |
| `--branch <int>` | Process a single file/branch | Great for quick local tests |
| `--print-status -1` | Show task tree + output existence | Use `-1` for full recursion |
| `--print-output <depth>` | Show output file paths at depth | `0` = the task itself |
| `--remove-output <depth>,<mode>[,y]` | Delete outputs | `a`=all, `i`=interactive, `d`=dry |
| `--workflow htcondor` | Submit to HTCondor | Requires `[job]` config in `law.cfg` |
| `--workflow slurm` | Submit to Slurm | Requires `[job]` config in `law.cfg` |
| `--help` | Full parameter list for a task | Use before every new task |

---

## GetDatasetLFNs

**What it does:** Resolves the list of ROOT file paths (Logical File Names) for a dataset. For CMS data it queries DAS using `dasgoclient`. Custom retrieval functions can be configured via `cfg.x.get_dataset_lfns`.

**Requires:** A valid GRID proxy (`voms-proxy-init -rfc -valid 196:00`) for CMS datasets, unless a custom LFN source is configured.

**Output:** A JSON file mapping file indices to LFN paths.

```bash
# Resolve LFNs for one dataset
law run cf.GetDatasetLFNs \
    --version dev1 \
    --dataset tt_dl_powheg

# Check which files were found
law run cf.GetDatasetLFNs --version dev1 --dataset tt_dl_powheg --print-output 0
```

**Tips:**
- Run this first for all datasets before starting the pipeline.
- If your datasets are stored locally or on a custom SE, configure `cfg.x.get_dataset_lfns` to bypass DAS.
- For debugging, add `--branch 0` to retrieve only the first file entry.

---

## CalibrateEvents

**What it does:** Applies one or more `Calibrator` objects to raw events, adding corrected column versions (e.g. corrected `Jet.pt` after JEC). The original columns are preserved.

**Outputs:** A Parquet file containing any new/modified columns declared in the calibrators' `produces` sets.

**Key parameters:**
- `--calibrators` — comma-separated list of calibrator names (order matters)
- `--shift` — which systematic shift variant to run (`nominal`, `jec_up`, `jec_down`, ...)

```bash
# Run default calibration on all files (triggers GetDatasetLFNs first)
law run cf.CalibrateEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --calibrators default

# Test on a single file first
law run cf.CalibrateEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --calibrators default \
    --branch 0

# Run with a JEC up-shift
law run cf.CalibrateEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --calibrators default \
    --shift jec_up

# Check if all branches completed
law run cf.CalibrateEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --print-status 0
```

**Tips:**
- For JEC uncertainties, separate task branches run for `nominal`, `jec_up`, and `jec_down` — this is handled automatically when you request a downstream task with multiple shifts.
- You rarely need to run `CalibrateEvents` standalone; it is triggered automatically by `SelectEvents`.

---

## SelectEvents

**What it does:** Runs the analysis `Selector` on calibrated events. Produces:
1. A Parquet file with event/object selection masks.
2. A `stats.json` file with event counts and MC weight sums (used for normalization in histogramming).
3. Optionally, selection-step histograms for cutflow plots.

Masks are **not yet applied** here — they are passed to `ReduceEvents`.

**Key parameters:**
- `--selector` — name of the exposed Selector to run
- `--calibrators` — calibrators to use upstream

```bash
# Run selector on all files of a dataset
law run cf.SelectEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --selector default \
    --calibrators default

# Inspect the full upstream task tree
law run cf.SelectEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --print-status -1

# Quick test: single file, no remote submission
law run cf.SelectEvents --version dev1 --dataset tt_dl_powheg --branch 0

# Run for a systematic shift (runs separate CalibrateEvents branch for jec_up)
law run cf.SelectEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --shift jec_up
```

**Tips:**
- After any change to the `Selector` code, bump the version to avoid reusing stale cached outputs.
- Check `stats.json` to verify event counts and selection efficiencies.
- The selector step names (keys in `SelectionResult.steps`) appear in cutflow plots.

---

## ReduceEvents

**What it does:** Applies the event and object masks from `SelectEvents` to all columns, writing a compact Parquet file with only the selected events. Columns not in `cfg.x.keep_columns["cf.ReduceEvents"]` are permanently dropped here.

**Output:** A Parquet file per input file with selected events and selected object collections.

```bash
# Reduce events for one dataset
law run cf.ReduceEvents \
    --version dev1 \
    --dataset tt_dl_powheg

# Single-file test
law run cf.ReduceEvents --version dev1 --dataset tt_dl_powheg --branch 0

# Check output size
law run cf.ReduceEvents --version dev1 --dataset tt_dl_powheg --print-output 0
```

**Tips:**
- Verify that all columns you need downstream are listed in `cfg.x.keep_columns["cf.ReduceEvents"]`.
- Use `ColumnCollection.ALL_FROM_SELECTOR` to automatically include all columns produced by your Selector.
- After changing `keep_columns`, remove and re-run `ReduceEvents`: `law run cf.ReduceEvents --version dev1 --remove-output 0,a,y`

---

## MergeReducedEvents

**What it does:** Merges the many per-file reduced Parquet files into larger files, targeting a configurable size (default 512 MB). This avoids having thousands of small files in downstream tasks.

```bash
law run cf.MergeReducedEvents \
    --version dev1 \
    --dataset tt_dl_powheg

# Adjust target file size (also configurable in cfg.x.reduced_file_size)
law run cf.MergeReducedEvents \
    --version dev1 \
    --dataset tt_dl_powheg \
    --merged-size 256
```

**Tips:**
- You rarely run this standalone; it is triggered automatically by `ProduceColumns` or `CreateHistograms`.

---

## ProduceColumns

**What it does:** Runs one or more `Producer` objects on the merged reduced events, adding new columns (high-level variables, weights, ML scores, category IDs). Each producer's output is stored in a separate Parquet file alongside the reduced events.

**Output:** A Parquet file per producer per dataset, containing the new columns.

**Key parameters:**
- `--producers` — comma-separated list of producer names (order matters: later producers can overwrite earlier columns of the same name)

```bash
# Run the default producer
law run cf.ProduceColumns \
    --version dev1 \
    --dataset tt_dl_powheg \
    --producers default

# Multiple producers (run in order; last one wins on name conflicts)
law run cf.ProduceColumns \
    --version dev1 \
    --dataset tt_dl_powheg \
    --producers default,extra_features

# Single-file test
law run cf.ProduceColumns \
    --version dev1 \
    --dataset tt_dl_powheg \
    --producers default \
    --branch 0
```

**Tips:**
- Producers can call other Producers; declare the sub-producers in `uses` and `produces`.
- Columns created in `ProduceColumns` do **not** need to be in `cfg.x.keep_columns`; they are automatically available downstream.
- If multiple producers write the same column name, the last producer in the `--producers` list wins.

---

## CreateHistograms

**What it does:** Fills `Hist` histograms for all requested variables, categories, and shifts. The `HistProducer` controls event weighting. One histogram file is produced per dataset branch.

**Key parameters:**
- `--variables` — comma-separated variable names (defined in config)
- `--categories` — comma-separated category names
- `--shift` — the systematic shift to use
- `--producers` — producers to run before histogramming
- `--hist-producer` — the HistProducer to use

```bash
# Basic: one variable, inclusive category, nominal shift
law run cf.CreateHistograms \
    --version dev1 \
    --dataset tt_dl_powheg \
    --variables jet1_pt \
    --categories incl \
    --shift nominal \
    --producers default

# Multiple variables and categories
law run cf.CreateHistograms \
    --version dev1 \
    --dataset tt_dl_powheg \
    --variables "jet1_pt,n_jet,ht,lep1_pt" \
    --categories "incl,sr,cr_1b"

# Run for all registered shifts at once (triggers separate branches per shift)
law run cf.CreateHistograms \
    --version dev1 \
    --datasets "tt_dl_powheg,wjets_madgraph" \
    --variables jet1_pt \
    --shifts "nominal,jec_up,jec_down,mu_up,mu_down"
```

**Tips:**
- Running with `--datasets "tt*"` processes all datasets matching the glob pattern.
- The `--shifts` flag (plural) allows specifying multiple shifts in one command; the task graph branches automatically.
- Use `--branch 0` to test on one file before submitting all branches.

---

## MergeHistograms

**What it does:** Merges per-branch histogram files into one file per dataset and shift. A second stage (`MergeShiftedHistograms`) further merges shifted histograms across all datasets of a process.

```bash
law run cf.MergeHistograms \
    --version dev1 \
    --variables jet1_pt \
    --processes tt,wjets,data_mu
```

**Tips:**
- This task is almost always triggered automatically by plotting or datacard tasks. You rarely call it standalone.

---

## PlotVariables1D

**What it does:** Creates 1D variable plots (stacked MC + data, or process-overlaid) from merged histograms using matplotlib/mplhep CMS style.

**Key parameters:**
- `--processes` — which processes to plot (alternative to `--datasets`)
- `--variables` — variables to plot
- `--categories` — categories to plot
- `--shift` — which shift to show
- `--file-types` — output format(s): `pdf`, `png`, `svg`
- `--skip-ratio` — disable the ratio panel below the main plot
- `--density` — normalize to unit area
- `--yscale` — `log` or `linear`
- `--view-cmd` — open plots automatically (e.g. `evince`, `eog`)

```bash
# Standard stacked plot: all tt + wjets + data
law run cf.PlotVariables1D \
    --version dev1 \
    --processes "tt,wjets,data_mu" \
    --variables "jet1_pt,n_jet,ht" \
    --categories incl \
    --shift nominal

# Save as both PDF and PNG
law run cf.PlotVariables1D \
    --version dev1 \
    --processes tt \
    --variables jet1_pt \
    --file-types "pdf,png"

# Log scale, skip ratio
law run cf.PlotVariables1D \
    --version dev1 \
    --processes "tt,wjets" \
    --variables jet1_pt \
    --yscale log \
    --skip-ratio

# Find where plots are saved
law run cf.PlotVariables1D \
    --version dev1 --processes tt --variables jet1_pt --print-output 0
```

---

## PlotVariables2D

```bash
law run cf.PlotVariables2D \
    --version dev1 \
    --processes tt \
    --variables "jet1_pt__jet1_eta" \
    --categories incl
```

---

## PlotShiftedVariables1D

**What it does:** Shows nominal and up/down shift variations for a variable, overlaid on one plot. Useful for visualizing the impact of a systematic on a distribution.

```bash
law run cf.PlotShiftedVariables1D \
    --version dev1 \
    --processes tt \
    --variables jet1_pt \
    --shift-sources "jec,mu" \
    --categories incl
```

---

## PlotCutflow

**What it does:** Bar chart showing the event yield after each selection step defined in the Selector (`SelectionResult.steps`).

```bash
law run cf.PlotCutflow \
    --version dev1 \
    --datasets tt_dl_powheg \
    --selector-steps "trigger,muon,jet,btag" \
    --categories incl
```

---

## PlotCutflowVariables1D

**What it does:** Plots a variable distribution at one or more intermediate selection steps (before the full selection is applied). Useful for gen-level checks or debugging.

```bash
# Gen-level top-quark pT at each selector step
law run cf.PlotCutflowVariables1D \
    --version dev1 \
    --datasets tt_dl_powheg \
    --processes tt \
    --variables genTop_pt \
    --categories incl \
    --skip-ratio
```

---

## CreateDatacards

**What it does:** Produces CMS `combine`-compatible datacards (`.txt`) and shape ROOT files from merged histograms, driven by an `InferenceModel` defined in your analysis.

```bash
law run cf.CreateDatacards \
    --version dev1 \
    --inference-model default \
    --variables jet1_pt \
    --categories sr
```

---

## Useful Workflow Commands

```bash
# --- Inspect ---
# Full task tree with output existence for a plot task
law run cf.PlotVariables1D --version dev1 --processes tt --variables jet1_pt --print-status -1

# Find output paths at depth 0 (the plot itself)
law run cf.PlotVariables1D --version dev1 --processes tt --variables jet1_pt --print-output 0

# --- Re-run ---
# Remove histogram task output and re-run it immediately (mode a = all)
law run cf.CreateHistograms --version dev1 --variables jet1_pt --remove-output 0,a,y

# --- Remote ---
# Submit SelectEvents to HTCondor
law run cf.SelectEvents \
    --version dev1 --dataset tt_dl_powheg --workflow htcondor

# --- Pin upstream versions ---
# Use prod1 outputs from ReduceEvents while creating new histograms as dev2
law run cf.CreateHistograms \
    --version dev2 \
    --cf.MergeReducedEvents-version prod1 \
    --variables jet1_pt

# --- Open columnar sandbox for ad-hoc scripting ---
cf_sandbox venv_columnar_dev bash
```

---

## Full Pipeline in One Command

Law resolves all upstream dependencies automatically. Running a plot task triggers the entire chain:

```bash
# This single command runs the full pipeline for one file of one dataset
law run cf.PlotVariables1D \
    --version dev1 \
    --datasets tt_dl_powheg \
    --processes tt \
    --variables jet1_pt \
    --categories incl \
    --branch 0
```

For production over all files and datasets, use HTCondor:

```bash
law run cf.MergeHistograms \
    --version prod1 \
    --datasets "tt_dl_powheg,wjets_madgraph,data_mu_b" \
    --variables "jet1_pt,n_jet,ht,lep1_pt" \
    --workflow htcondor
```
