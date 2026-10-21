#!/usr/bin/env bash

# Copies the outputs of the given task families out of a columnflow store into a destination
# directory, one folder per task family. The e2e workflows upload that directory as a small
# "results" artifact holding the plots (and tables, datacards) of exactly the tasks a job ran,
# rather than the whole store with all intermediate outputs (uploaded on failure only).
#
# Usage: bash collect_results.sh STORE DEST FAMILY [FAMILY ...]
#   e.g. bash collect_results.sh citest/data/cf_store "${RUNNER_TEMP}/results" cf.PlotVariables1D
#
# The store is laid out as STORE/<analysis>/<task family>/..., the results as DEST/<task family>/...
# A family without outputs is only reported as a warning: the job's own checks decide whether
# outputs are missing, and on a failed job the outputs collected so far are still worth uploading.

action() {
    local store="$1"
    local dest="$2"
    shift 2
    if [ -z "${store}" ] || [ -z "${dest}" ] || [ "$#" = "0" ]; then
        >&2 echo "usage: collect_results.sh STORE DEST FAMILY [FAMILY ...]"
        return 1
    fi

    mkdir -p "${dest}" || return "$?"
    local family dir found
    for family in "$@"; do
        found="false"
        for dir in "${store}"/*/"${family}"; do
            [ -d "${dir}" ] || continue
            mkdir -p "${dest}/${family}" && cp -r "${dir}/." "${dest}/${family}/" || return "$?"
            found="true"
        done
        if ${found}; then
            echo "collected ${family}: $( find "${dest}/${family}" -type f | wc -l ) files"
        else
            echo "::warning::no outputs of ${family} found in ${store}"
        fi
    done
}
action "$@"
