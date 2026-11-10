#!/usr/bin/env bash

# Sourced by every law-running step of .github/workflows/template_e2e.yaml, from the generated
# analysis directory ($CF_ANALYSIS_DIR), in place of a bare "source setup.sh". It sets up the
# analysis environment and then re-declares the runner as a *local* columnflow environment.
#
# Why the override is needed:
# setup.sh's cf_detect_envs() maps GITHUB_ACTIONS=true to CF_CI_ENV=true and hence to
# CF_LOCAL_ENV=false (setup.sh:238-257). columnflow reads that flag once at import time into
# columnflow.env_is_local, and two tasks of the chain refuse to run unless it is true:
#
#   - cf.GetDatasetLFNs.run is decorated with @only_local_env (columnflow/tasks/external.py:94),
#     raising "cf.GetDatasetLFNs.run() can only be executed locally", and
#   - cf.BundleExternalFiles.run refuses to create a missing bundle
#     (columnflow/tasks/external.py:646-651), raising "the output bundle ... is missing, but
#     cannot be created in non-local environments".
#
# Both guards exist so that remote/CI jobs never reach out to DAS or the grid themselves. Here
# neither task does: the CI overlay (tests/ci/ci_config_patch.py) has already redirected LFN
# resolution and all correctionlib inputs to the local $CF_CI_TESTDATA bundle, so the runner
# genuinely is the local environment steering this run.
#
# The override is applied *after* sourcing, deliberately: while sourcing, CF_LOCAL_ENV also gates
# the conda/micromamba installation and the build of the "cf" production venv
# (setup.sh:641,745) - neither is needed here, and installing conda would only slow the job down.
# Leaving the flag false during sourcing preserves that, and flips only the runtime task guards.
#
# Sandboxed tasks (everything from cf.CalibrateEvents onwards runs in venv_columnar) re-source
# setup.sh in their subshell, but do NOT see CF_LOCAL_ENV=false again: sandboxes/_setup_venv.sh:62
# sources it as `CF_SKIP_SETUP="true" source .../setup.sh ""`, and setup.sh's entry point is
# `if ! ${CF_SKIP_SETUP}; then main "$@"; fi` (setup.sh:1181), so main() - and with it
# cf_detect_envs() (setup.sh:238-257), the function that would reset CF_LOCAL_ENV - never runs
# inside a sandbox. CF_LOCAL_ENV is therefore simply inherited from this script's export, i.e. it
# is "true" in every sandboxed subprocess too. That is harmless here regardless, because the only
# consumers of columnflow.env_is_local are:
#   - columnflow/tasks/external.py: the @only_local_env decorator on GetDatasetLFNs.run (line 94)
#     and the bundle-missing guard in BundleExternalFiles.run (line 647), both unsandboxed tasks
#     this workflow already accounts for above, and
#   - five @only_local_env-decorated remote-submission methods in
#     columnflow/tasks/framework/remote.py (BundleGitRepository.run, BundleSoftware.run,
#     BuildBashSandbox.run, BundleBashSandbox.run, BundleCMSSWSandbox.run), none of which this
#     workflow ever triggers since it never submits remote jobs.

source setup.sh "" || return "$?"

export CF_LOCAL_ENV="true"
