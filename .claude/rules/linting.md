# Linting

After writing or editing any Python file, run flake8 and fix all reported errors before considering the task done.

## Command

```bash
flake8 --config=.flake8 <file_or_directory>
```

Example:
```bash
flake8 --config=.flake8 hbw/tasks/trigger_sf.py
flake8 --config=.flake8 hbw/
```

## Project settings (from `.flake8`)

- `max-line-length = 120`
- `ignore = E128, E306, E402, E722, E731, W504, Q003`
- `inline-quotes = double` — use `"..."` not `'...'` for strings

## Workflow

1. Write or edit the Python code.
2. Run `flake8 --config=.flake8 <file>`.
3. Fix every reported error.
4. Re-run flake8 to confirm zero errors.
5. Only then consider the code complete.

## Common fixes

- **E501** (line too long): break at operator, wrap in parentheses, or shorten names
- **F401** (unused import): remove the import
- **E711** (comparison to None): use `is None` / `is not None`, not `== None`
- **W291/W293** (trailing whitespace): remove trailing spaces
- **Q000** (wrong quote type): replace `'...'` with `"..."`
