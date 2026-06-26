# Columnflow Pipeline

## Standard task order

```
GetDatasetLFNs → CalibrateEvents → SelectEvents → ReduceEvents
→ MergeReducedEvents → ProduceColumns → CreateHistograms
→ MergeHistograms → PlotVariables1D / CreateDatacards
```

Law resolves upstream dependencies automatically. Running a downstream task triggers all missing upstream tasks.

## Five TAF types

| TAF | Class | Task | CLI flag | Count |
|---|---|---|---|---|
| Calibrator | `Calibrator` | `CalibrateEvents` | `--calibrators` | 0..N |
| Selector | `Selector` | `SelectEvents` | `--selector` | exactly 1 |
| Reducer | `Reducer` | `ReduceEvents` | `--reducer` | exactly 1 |
| Producer | `Producer` | `ProduceColumns` | `--producers` | 0..N |
| HistProducer | `HistProducer` | `CreateHistograms` | `--hist-producer` | exactly 1 |

## What each task produces

- **CalibrateEvents** → Parquet with additional/corrected columns
- **SelectEvents** → Parquet with event/object masks + `stats.json` (event counts, MC weight sums)
- **ReduceEvents** → Parquet with selected events only; columns not in `cfg.x.keep_columns["cf.ReduceEvents"]` are permanently dropped
- **ProduceColumns** → Parquet with new columns alongside the reduced events
- **CreateHistograms** → pickle with `Hist` histograms per dataset/shift/category

## Role of each step (how you use it)

- **CalibrateEvents** — calibrate/correct physics objects (JEC/JER, electron scale & smear, …).
- **SelectEvents** — apply the *event-level* selection that reduces how many events flow
  downstream. Keep it to the cuts that genuinely remove events; finer object-level or
  region splits belong in `ProduceColumns` categories (see below).
- **ReduceEvents** — drop unselected events and all columns not in
  `cfg.x.keep_columns["cf.ReduceEvents"]`. Keep columns **generously**: it is cheaper to save
  a column you might not need than to re-run the whole chain because one was dropped.
- **ProduceColumns** — produce every derived column needed downstream, the categories that act
  as **additional selections layered on top of the Selector** (region/working-point splits
  applied at histogram time, not by dropping events), and all Monte-Carlo event weights
  (normalization, scale factors, pileup).
- **CreateHistograms** — pick the `--hist-producer` by what the histogram is for. The standard
  one applies all event weights for MC and weight 1 for data; write a custom HistProducer only
  when a task needs non-standard histogram contents.
- **PlotVariables1D / PlotVariables2D** — consume the histograms; the primary feedback loop for
  understanding a region and validating the selections/categories applied upstream.

## law.cfg: module registration

Every new Python file containing a TAF must be added to `law.cfg`:

```ini
calibration_modules:   myanalysis.calibration.{default,jets}
selection_modules:     myanalysis.selection.{default,objects}
production_modules:    myanalysis.production.{default,weights}
categorization_modules: myanalysis.categorization.categories
hist_production_modules: myanalysis.histogramming.default
inference_modules:     myanalysis.inference.default
```

No spaces after commas inside `{}` brace expansions.
