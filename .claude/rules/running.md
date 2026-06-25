# Running Columnflow Tasks

Operational reference for executing the pipeline (testing a TAF, reproducing a run,
clearing stale output). For task *order* and TAF types see `01-pipeline.md`.

## Environment — required before any `law` command

```bash
source setup.sh default
```

Never call `law`, bare `python`, or `pip` without sourcing first. A missing-package
error after sourcing means the package is absent from the columnar venv, not that the
env failed to load.

## Anatomy of a run command

```bash
law run cf.<Task> --version <ver> --dataset <ds> <taf-flag> [--branch N] [run-control flags]
```

Per-task-type flags (exactly one TAF flag where the count is "exactly 1"):

| Task | TAF flag |
|---|---|
| `cf.CalibrateEvents` | `--calibrators <c1,c2>` (0..N, omit for none) |
| `cf.SelectEvents` | `--selector <name>` |
| `cf.ReduceEvents` | `--reducer <name>` |
| `cf.ProduceColumns` | `--producers <p1,p2>` |
| `cf.CreateHistograms` | `--hist-producer <name>` |

Law resolves upstream automatically: running a downstream task triggers every missing
upstream task. You do **not** need to run the chain by hand — running
`cf.ProduceColumns --branch 0` builds calibrate → select → reduce → produce for branch 0.

## Testing a single chunk — always `--branch 0` first

```bash
law run cf.SelectEvents --version dev1 --dataset tt_sl_powheg --selector my_sel --branch 0
```

`--branch 0` runs only the first file chunk (limited statistics) — the standard smoke
test. Never submit a full dataset (all branches) before branch 0 succeeds locally.

## Checking status without running

```bash
law run cf.<Task> <flags> --print-status -1
```

`--print-status DEPTH`: `-1` walks the full upstream dependency tree; `0` shows only the
target task. Each output line is marked **existent** (present) or **absent** (missing).
Use this before a run (confirm the task is registered and what's already built) and after
a run (confirm output is present). If the task name itself is rejected, the TAF is not
registered in `law.cfg` — fix registration, do not retry verbatim.

## Re-running — clear stale output first

```bash
law run cf.<Task> <flags> --remove-output 0,a,y
```

`--remove-output DEPTH,MODE,APPROVAL`:

- **DEPTH** `0` = this task only; higher = also remove upstream outputs.
- **MODE** `a` = remove all matching output. (`i` interactive, `d` dry-run/list-only.)
- **APPROVAL** `y` = skip the confirmation prompt.

`0,a,y` (clear this task's output, no prompt) is the usual re-run incantation. Removing
output then re-running is the correct way to pick up a code change — law caches by output
existence, so an unchanged-looking output is **not** re-made unless removed.

## Sandboxes (first-run slowness is normal)

TAFs run inside sandboxes (e.g. `venv_columnar.sh`). The **first** invocation that needs a
given sandbox builds it, which can take minutes and prints pip/build output. This is setup,
not a task failure — let it finish; subsequent runs reuse the built venv.

## Scale-out (not for smoke tests)

```bash
law run cf.<Task> <flags> --workflow htcondor      # submit all branches to the batch
law run cf.<Task> <flags> --workers 4              # local parallelism across branches
```

Default (no `--workflow`) runs locally. Smoke tests stay local, `--branch 0`, no extra
workers.

## Reading a failure

| Symptom | Likely cause | Action |
|---|---|---|
| Task name not recognized by `law` | TAF module not in `law.cfg` `*_modules` | fix registration (see `01-pipeline.md`) |
| `ImportError` / `ModuleNotFoundError` | bad import / package absent from venv | fix import; verify package is in the columnar venv |
| `AttributeError` on `events.<X>` at runtime | column read but not in `uses`, or dropped at Reduce | add to `uses`; if pre-Reduce, add to `keep_columns` (see `02-invariants.md`) |
| awkward `ValueError` / type/broadcast error | shape or `axis` mismatch in computation | inspect the computation (logic bug) |
| output target / `LocalFileTarget` error | output config or permissions | check `law.cfg` outputs / store paths |

First-run sandbox build time and `--print-status` "absent" lines for not-yet-built
upstream are **not** failures.
