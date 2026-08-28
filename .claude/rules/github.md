# GitHub / Git Behaviour

The remote is **GitLab** (`ssh://git@gitlab.cern.ch:7999/juvanden/cffakerates.git`), so a "pull
request" here is a **merge request (MR)**. `gh` is on `PATH` but does not work against this remote —
never use it. MRs are created with `git push -o merge_request.create` (see below).

## Branch model

```text
master                          reference only — never run from, never merged into by the agent
 └── fakerate_2024_jfc_claude   ANALYSIS branch: the user runs law tasks from this checkout
      ├── task/T18-<slug>       one per JFC task, developed in its own worktree
      └── task/T19-<slug>
```

- `fakerate_2024_jfc_claude` is the **analysis branch** — the state the user runs the analysis
  from. It must stay runnable at all times.
- `master` is not used: never run from it, never open an MR into it, never merge it.
- **One branch per JFC task**, named `task/T<N>-<kebab-slug>` with the `T<N>` from
  `IMPLEMENTATION_STRATEGY.md` (e.g. `task/T18-closure-plots`).
- Task branches always start from the current tip of `fakerate_2024_jfc_claude`.
- Never commit task work directly to `fakerate_2024_jfc_claude`. The one exception is quick mode
  (`QUICKFIX:` prefix, see `.claude/jfc/orchestrator.md`), which commits straight to the analysis
  branch — no task branch, no worktree.

## Task worktree

Every task is developed in its **own git worktree**, so the main checkout stays on
`fakerate_2024_jfc_claude` and the user can keep running `law` tasks while a task is in flight.
Create it at the REFRESH step of the task loop, before any executor runs:

```bash
git worktree add -b task/T18-closure-plots ../cfFakeRates-wt/T18-closure-plots fakerate_2024_jfc_claude
```

Bootstrap it so it **shares software and store** with the main checkout. Run this in a **fresh
shell** — the main checkout's setup exports `CF_*`, `CFFAKERATES_*` and `LAW_*` that otherwise leak
in and point law back at the main checkout:

```bash
cd ../cfFakeRates-wt/T18-closure-plots
MAIN=/ada_mnt/ada/user/juvanden/columnflowProjects/cfFakeRates

# submodules must exist BEFORE setup.sh runs — it sources modules/columnflow/setup.sh
# symlinking to the main checkout costs no clone and keeps its local columnflow modifications
rmdir modules/columnflow modules/cmsdb
ln -s "$MAIN/modules/columnflow" modules/columnflow
ln -s "$MAIN/modules/cmsdb" modules/cmsdb

export CF_DATA="$MAIN/data"     # must be set BEFORE sourcing setup.sh
source setup.sh default
```

Expected result (~40 s, no venv build): `CF_VENV_BASE` and `CF_STORE_LOCAL` under the main
checkout's `data/`, while `LAW_HOME`, `LAW_CONFIG_FILE` and `PYTHONPATH` point at the worktree.

Rules:

- Worktrees live under `../cfFakeRates-wt/<T-slug>` — **outside** the repo, so they never show up in
  `git status` and cannot be swept up by a stray `git add`.
- `CF_DATA` must be exported **before** sourcing `setup.sh`. `setup.sh` derives `CFFAKERATES_BASE`
  from its own directory, and columnflow derives `CF_SOFTWARE_BASE` / `CF_VENV_BASE` /
  `CF_STORE_LOCAL` from `CF_DATA` (default `$CFFAKERATES_BASE/data`). Without the export the
  worktree builds its own venvs from scratch and starts an empty store.
- `data/`, `.law/` and `.setups/` are gitignored, so a fresh worktree has none of them; `LAW_HOME`
  and `LAW_CONFIG_FILE` come from the worktree itself, which is correct and cheap.
- A new worktree starts with **empty** `modules/*`, and `setup.sh` sources
  `modules/columnflow/setup.sh` before its own submodule-init loop — so the symlinks (or a
  `git submodule update --init --recursive`) must come first, or the setup aborts. Prefer the
  symlinks: a fresh clone would drop the main checkout's uncommitted columnflow modifications that
  the analysis currently depends on.
- A worktree contains only **committed** state. Uncommitted work in the main checkout (new configs,
  edited TAFs) is not there — commit it to the analysis branch first if a task needs it.
- **Always use a task-scoped `--version` from a worktree.** The store path only separates outputs by
  TAF names and version
  (`.../<cf.Task>/<config>/<dataset>/<shift>/calib__*/sel__*/prod__*/<version>/`), so the same
  producer name at the same version with changed code silently overwrites what the user's checkout
  depends on:

  ```bash
  law run cf.SelectEvents --version t18 --dataset tt_sl_powheg --branch 0
  ```

- To avoid re-running unchanged upstream in the new version, pin upstream tasks to the existing
  version with task-family-prefixed parameters:

  ```bash
  law run cf.ProduceColumns --version t18 \
      --cf.SelectEvents-version dev3 --cf.ReduceEvents-version dev3 --branch 0
  ```

- Never `git checkout` / `git switch` in the **main** checkout — it must stay on
  `fakerate_2024_jfc_claude`. That is the entire point of the worktree.
- Clean up (`git worktree remove`, `git worktree prune`) only after the MR is merged **and** the user
  asks. Never automatically, never with `--force`.

## Merge requests

Open the MR only after the task is committed and **every applicable gate has passed**
(fixer → tester → physics). Target is always the analysis branch:

```bash
git push -u origin task/T18-closure-plots \
  -o merge_request.create \
  -o merge_request.target=fakerate_2024_jfc_claude \
  -o merge_request.title="T18: closure plots"
```

- Pushing a `task/*` branch and opening its MR is **pre-authorized** — no confirmation needed.
- Pushing to `fakerate_2024_jfc_claude` or `master` is **not** — ask first.
- The MR description links the review and test artifacts:
  `.claude/jfc/reviews/<task_slug>-fixer.md`, `<task_slug>-physics.md`, `.claude/jfc/tests/<task_slug>.md`.
- **Never merge the MR.** Merging is the user's decision, always.
- Never open an MR into `master`. If asked to, refuse and explain that `fakerate_2024_jfc_claude` is
  the integration target.

## Summary table

| Action | Allowed |
|---|---|
| `git commit` on a `task/*` branch, inside its worktree | Yes |
| `git worktree add` for a task | Yes |
| `git push` of a `task/*` branch + open its MR (all gates passed) | Yes — pre-authorized |
| `git push` to `fakerate_2024_jfc_claude` or `master` | Ask first |
| `git commit` to `fakerate_2024_jfc_claude` | Only in quick mode (`QUICKFIX:`) |
| Merging an MR | Never — the user merges |
| `git rebase` | Ask first |
| `git checkout` / `git switch` in the main checkout | Never |
| `git branch -d` / `-D` | Never |
| `git worktree remove --force`, `git reset --hard`, force push | Never |
| MR: `task/*` → `fakerate_2024_jfc_claude` | Yes |
| MR/PR: anything → `master` | Never |

Stage specific files only — never `git add -A` or `git add .`.

## Anti-patterns

- Committing task work on `fakerate_2024_jfc_claude` instead of a `task/*` branch
- Creating a worktree without exporting `CF_DATA` first (rebuilds venvs, empty store)
- Running `law` from a worktree without a task-scoped `--version` (overwrites the user's outputs)
- Switching the main checkout to another branch instead of using a worktree
- Removing a worktree or deleting its branch before the MR is merged
- Using `gh` against this GitLab remote
- Merging an MR yourself
