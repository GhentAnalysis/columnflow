# Linting

## While writing code

Always respect the `.flake8` settings when writing code — do not produce code that violates them:

- `max-line-length = 120`
- `ignore = E128, E306, E402, E722, E731, W504, Q003`
- `inline-quotes = double` — use `"..."` not `'...'` for strings
- Never leave active `breakpoint()` calls (CI scans for them — add `# noqa` if intentional)

## Before pushing to GitHub / GitLab

Run all checks below and fix any failures. Commit fixes separately with the message `linting fixes`.

All commands require the columnflow environment (`source setup.sh default`).

### 1. Python linting — `bash tests/run_linting`

Runs `flake8 columnflow tests bin docs setup.py`. Must be clean before pushing.

```bash
bash tests/run_linting
# or for a single file:
flake8 --config=.flake8 <file>
```

### 2. Markdown linting — `bash tests/run_docs lint`

Runs `pymarkdown` on `README.md` and `docs/`. Rules configured in `.markdownlint`:

- MD031: fenced code blocks must be surrounded by blank lines
- MD032: lists must be surrounded by blank lines
- MD040: fenced code blocks must have a language tag (use `text` for diagrams/plain text)

```bash
bash tests/run_docs lint
```

### 3. Breakpoint check (CI only — no local script)

CI scans for uncommented `breakpoint()` calls. Do not leave them in pushed code.

---

## CI jobs on push / pull request

| Job | Script | What it checks |
|---|---|---|
| `lint` | `tests/run_linting` | flake8 on `columnflow/`, `tests/`, `bin/`, `docs/`, `setup.py` |
| `lint_docs` | `tests/run_docs lint` | pymarkdown on `README.md` and `docs/` |
| `breaks` | inline `git grep` | no active `breakpoint()` calls |
| `test` | `tests/run_tests` | all unit tests (see below) |
| `pypi` | `python setup.py sdist` + `twine check` | package builds and passes PyPI checks |
| `coverage` | `tests/run_coverage` | coverage upload to Codecov |

---

## Unit tests

### Run all tests

```bash
bash tests/run_tests
```

### Tests without a sandbox (fast)

```bash
python -m unittest tests.test_util tests.test_task_parameters tests.test_base_tasks
```

### Tests requiring the columnar venv sandbox

```bash
bash tests/run_test test_columnar_util sandboxes/venv_columnar.sh
bash tests/run_test test_config_util   sandboxes/venv_columnar.sh
bash tests/run_test test_inference     sandboxes/venv_columnar.sh
bash tests/run_test test_hist_util     sandboxes/venv_columnar.sh
bash tests/run_test test_plotting      sandboxes/venv_columnar.sh
```

### What each test module covers

| Module | Sandbox | Covers |
|---|---|---|
| `test_util` | no | `maybe_import`, utility functions (`save_div`, `try_int`, `is_regex`, ...) |
| `test_columnar_util` | yes | `Route` class: join, split, apply, tags |
| `test_config_util` | yes | `get_events_from_categories` and config utilities |
| `test_inference` | yes | `InferenceModel` specs: process, category, parameter, parameter groups |
| `test_hist_util` | yes | `create_hist_from_variables`, `translate_hist_intcat_to_strcat` |
| `test_task_parameters` | no | `SettingsParameter`, `MultiSettingsParameter` |
| `test_base_tasks` | no | `AnalysisTask`: resolve config, categories, variables, datasets, shifts |
| `test_plotting` | yes | `PlotUtil`, confusion matrix, ROC curve plot utilities |

> `test_selectionresults.py` is a standalone manual script, not a unittest — it is not run by `run_tests`.

### Workflow

1. Run the no-sandbox tests first for a quick check.
2. If changes touch columnar utilities, histogramming, inference, or plotting, run `bash tests/run_tests`.
3. Fix any failures before pushing.
