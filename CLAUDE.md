# Columnflow Project

Backend for columnar, fully orchestrated HEP analyses with [law](https://github.com/riga/law) (workflow orchestration), [order](https://github.com/riga/order) (metadata), and [awkward-array](https://awkward-array.org) (columnar event data).

## Always-loaded rules (in `.claude/rules/`)

- **01-pipeline.md** — task order, 5 TAF types, law.cfg module registration
- **02-invariants.md** — set_ak_column, uses/produces, imports, Selector return type, MC guard, keep_columns, no loops

Path-scoped rules load automatically when Claude works with files in the corresponding directory:
`selection/`, `production/`, `calibration/`, `config/`, `categorization/`, `histogramming/`, `inference/`

## Commands

```bash
law run cf.<TaskName> --version <name> [--config <cfg>] [--dataset <name>] [--branch 0]
law run cf.PlotVariables1D --version dev1 --processes tt --variables jet1_pt --print-status -1
law run cf.SelectEvents    --version dev1 --dataset tt_dl_powheg --remove-output 0,a,y
```

## User guide

Full documentation for users is in `docs/columnflow_claude_guide/`:
`01_framework_structure.md` · `02_coding_style.md` · `03_task_commands.md` · `04_awkward_reference.md` · `05_dos_and_donts.md` · `06_custom_tasks.md`
