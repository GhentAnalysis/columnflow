# GitHub / Git Behaviour

## Allowed (no confirmation needed)

- Commit to the current branch (any branch, including main)
- Create a new branch for feature development

## Not allowed — stop and ask the user first

- **Push to any branch** — always ask for confirmation before pushing, even if the user says "push it"
- **Rebase** — never rebase without explicit permission for that specific rebase
- **Delete a branch or commit** — never delete; warn the user if they request it

## Pull request rules

- **Never create a PR from a feature branch directly to main/master.** An intermediate integration branch must always exist between the feature branch and main.
- Branch hierarchy: `feature/* → integration/staging branch → main`
- If asked to open a PR to main directly, refuse and explain that an intermediate branch is required.

## Summary table

| Action | Allowed |
|---|---|
| `git commit` | Yes |
| `git checkout -b <new-branch>` | Yes |
| `git push` | Ask first |
| `git rebase` | Ask first |
| `git branch -d` / `git branch -D` | Never |
| `git reset --hard` / force operations | Never |
| PR: feature → main | Never (use intermediate branch) |
| PR: feature → intermediate | Yes, after push confirmation |
