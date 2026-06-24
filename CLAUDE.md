# Columnflow Project — Claude Instructions

This is a **columnflow** analysis project. Columnflow is a fully orchestrated columnar HEP analysis framework built on [law](https://github.com/riga/law) (workflow orchestration) and [order](https://github.com/riga/order) (metadata management), using [awkward-array](https://awkward-array.org) for columnar event data.

Read all five guide files before writing or modifying any code:

| Guide | Contents |
|---|---|
| [01 Framework Structure](docs/columnflow_claude_guide/01_framework_structure.md) | Analysis pipeline, task graph, config objects (Analysis/Campaign/Config), Order objects |
| [02 Coding Style](docs/columnflow_claude_guide/02_coding_style.md) | TAF decorators, uses/produces, imports, module registration, naming conventions |
| [03 Task Commands](docs/columnflow_claude_guide/03_task_commands.md) | `law run` commands with all standard parameters for every task |
| [04 Awkward Reference](docs/columnflow_claude_guide/04_awkward_reference.md) | Key `ak.*` functions used in columnflow code |
| [05 Dos and Don'ts](docs/columnflow_claude_guide/05_dos_and_donts.md) | Explicit rules Claude must follow when writing columnflow code |

## Quick orientation

- All data operations act on **chunks** of events encoded as `ak.Array` named `events`.
- Task array functions (TAFs): `Calibrator`, `Selector`, `Producer`, `Reducer`, `HistProducer`.
- Each TAF declares `uses` (columns to read) and `produces` (columns to write) in its decorator.
- Events are **never modified in-place** — always use `set_ak_column(events, "FieldName", value)` and reassign `events`.
- The standard pipeline order: `GetDatasetLFNs → CalibrateEvents → SelectEvents → ReduceEvents → ProduceColumns → CreateHistograms → PlotVariables1D`.
- All tasks are run with `law run cf.<TaskName> --version <name> [--config <cfg>] [...]`.
- New Python modules must be registered in `law.cfg` under the correct `*_modules` key.
