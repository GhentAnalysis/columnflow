# Linting

## While writing code

Always respect the `.flake8` settings when writing code — do not produce code that violates them:

- `max-line-length = 120`
- `ignore = E128, E306, E402, E722, E731, W504, Q003`
- `inline-quotes = double` — use `"..."` not `'...'` for strings

## Before pushing to GitHub / GitLab

Run flake8 and fix all errors before pushing. Commit the fixes separately with the message `linting fixes`.

### Prerequisites

`flake8` requires the columnflow environment:

```bash
source setup.sh default
```

### Commands

```bash
# Full project lint (canonical — mirrors CI)
bash tests/run_linting

# Single file or directory
flake8 --config=.flake8 <file_or_directory>
```

### Workflow

1. Finish the feature/fix and commit it.
2. Run `bash tests/run_linting`.
3. If errors are reported: fix them, then commit with message `linting fixes`.
4. Push only when linting is clean.

### Common fixes

- **E501** (line too long): break at operator, wrap in parentheses, or shorten names
- **F401** (unused import): remove the import
- **E711** (comparison to None): use `is None` / `is not None`, not `== None`
- **W291/W293** (trailing whitespace): remove trailing spaces
- **Q000** (wrong quote type): replace `'...'` with `"..."`

---

# Tests

After writing code that touches columnflow core, run the unit tests.

## Prerequisites

Same environment: `source setup.sh default`

## Commands

```bash
# All tests (some require the columnar sandbox — may be slow)
bash tests/run_tests

# Tests that run without a sandbox (fast, always available)
python -m unittest tests.test_util tests.test_task_parameters tests.test_base_tasks

# Tests that need the columnar venv sandbox
bash tests/run_test test_columnar_util sandboxes/venv_columnar.sh
bash tests/run_test test_config_util   sandboxes/venv_columnar.sh
bash tests/run_test test_inference     sandboxes/venv_columnar.sh
bash tests/run_test test_hist_util     sandboxes/venv_columnar.sh
bash tests/run_test test_plotting      sandboxes/venv_columnar.sh
```

## Workflow

1. Run `python -m unittest tests.test_util tests.test_task_parameters tests.test_base_tasks` for a quick sanity check (no sandbox needed).
2. If changes touch columnar utilities, histogramming, inference, or plotting, run the full `bash tests/run_tests`.
3. Fix any failures before pushing.
