#!/usr/bin/env bash

# Helpers shared by the test-template e2e scripts in this directory (see tests/ci/README.md, section
# "Test template"). Sourced by each script from the generated analysis directory, it sets up the
# analysis environment via tests/ci/setup_ci_env.sh and defines:
#
#   cf_law <task> [args...]          runs "law run <task> ${TASK_ARGS} [args...]" in a log group
#   find_outputs <family> <name>     prints store files of task family <family> named <name> (glob)
#   find_one <family> <name> [path]  like find_outputs, optionally also matching the path glob
#                                    <path>, but fails unless exactly one file is found
#   assert_some <family> <name>      fails unless at least one matching file exists, prints them
#   check_py <script> [args...]      runs the python check <script> of this directory in the
#                                    columnar sandbox (needed to unpickle hists, read root files)
#
# Variables that can be preset by the caller (the workflow sets them as job env):
#   TASK_ARGS    arguments passed to every law run, default "--version ci --config l18 ..."
#   CF_E2E_DATASET   the dataset to process, default "tt_dl_powheg"

CF_E2E_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CF_E2E_REPO="$( cd "${CF_E2E_DIR}/../../.." && pwd )"

# setup the analysis, declaring the runner a local environment, and index tasks explicitly as the
# automatic indexing while sourcing is skipped in CI (see tests/ci/setup_ci_env.sh)
source "${CF_E2E_REPO}/tests/ci/setup_ci_env.sh" || exit "$?"
law index -q || exit "$?"

TASK_ARGS="${TASK_ARGS:---version ci --config l18 --workers 1 --local-scheduler True}"
CF_E2E_DATASET="${CF_E2E_DATASET:-tt_dl_powheg}"

# store of all local outputs, defined by the analysis setup
CF_E2E_STORE="${CF_STORE_LOCAL:?CF_STORE_LOCAL not set after the analysis setup}"

_cf_e2e_group() {
    if [ "${GITHUB_ACTIONS}" = "true" ]; then
        echo "::group::$*"
    else
        echo -e "\n>>> $*"
    fi
}

_cf_e2e_endgroup() {
    [ "${GITHUB_ACTIONS}" = "true" ] && echo "::endgroup::"
    return "0"
}

_cf_e2e_error() {
    if [ "${GITHUB_ACTIONS}" = "true" ]; then
        echo "::error::$*"
    else
        >&2 echo "ERROR: $*"
    fi
}

cf_law() {
    local task="$1"
    shift
    _cf_e2e_group "law run ${task} $*"
    # TASK_ARGS is deliberately unquoted to split it into separate arguments
    law run "${task}" ${TASK_ARGS} "$@"
    local ret="$?"
    _cf_e2e_endgroup
    if [ "${ret}" != "0" ]; then
        _cf_e2e_error "law run ${task} $* failed with exit code ${ret}"
    fi
    return "${ret}"
}

find_outputs() {
    local family="$1"
    local name="$2"
    local path_glob="${3:-*}"
    find "${CF_E2E_STORE}" -path "*/${family}/*" -path "${path_glob}" -name "${name}" -type f | sort
}

find_one() {
    local matches=()
    mapfile -t matches < <( find_outputs "$@" )
    if [ "${#matches[@]}" != "1" ]; then
        _cf_e2e_error "expected exactly one '$2' output of $1 (path '${3:-*}'), found ${#matches[@]}"
        printf ' - %s\n' "${matches[@]}" >&2
        return "1"
    fi
    echo "${matches[0]}"
}

assert_some() {
    local matches=()
    mapfile -t matches < <( find_outputs "$@" )
    if [ "${#matches[@]}" = "0" ]; then
        _cf_e2e_error "no '$2' output of $1 found (path '${3:-*}')"
        return "1"
    fi
    echo "found ${#matches[@]} '$2' output(s) of $1:"
    printf ' - %s\n' "${matches[@]}"
}

check_py() {
    local script="$1"
    shift
    _cf_e2e_group "check ${script} $*"
    # cf_sandbox joins its arguments into a single bash -c string, so all arguments must be free of
    # spaces, which holds for store paths and the plain values passed by the e2e scripts
    cf_sandbox venv_columnar python "${CF_E2E_DIR}/${script}" "$@"
    local ret="$?"
    _cf_e2e_endgroup
    if [ "${ret}" != "0" ]; then
        _cf_e2e_error "check ${script} failed with exit code ${ret}"
    fi
    return "${ret}"
}
