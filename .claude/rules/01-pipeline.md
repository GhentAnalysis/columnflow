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
