#!/usr/bin/env bash

# Base job of the test-template e2e workflow: runs the part of the chain that every feature job
# shares (calibration, selection, reduction and the b-tag efficiencies needed by the event weight
# producers). Its store is handed to the feature jobs, which then find these outputs complete.
# Run from the generated analysis directory.

source "$( dirname "${BASH_SOURCE[0]}" )/lib.sh" || exit "$?"

action() {
    local ds="--dataset ${CF_E2E_DATASET}"

    cf_law cf.CalibrateEvents ${ds} || return "$?"
    cf_law cf.SelectEvents ${ds} || return "$?"

    # merged selection stats, checked against the 66000 events of the bundled file that
    # tests/ci/ci_config_patch.py serves N_LFN_REPEATS (= 2) times; the efficiency is unaffected by
    # the repetition (observed ~0.166 historically)
    cf_law cf.MergeSelectionStats ${ds} || return "$?"
    local stats
    stats="$( find_one cf.MergeSelectionStats stats.json )" || return "$?"
    python3 "${CF_E2E_DIR}/check_stats.py" "${stats}" 132000 0.10 0.25 || return "$?"

    # reduction, including the merging step that is only run for datasets with more than one file
    cf_law cf.ReduceEvents ${ds} || return "$?"
    cf_law cf.MergeReductionStats ${ds} || return "$?"
    cf_law cf.MergeReducedEvents ${ds} || return "$?"

    # b-tag efficiencies, required by fixed_wp_btag_weights in the event_weights producer (the
    # plot task requires the efficiency task, with the same parameters as the producer does); the
    # variables are passed explicitly like the producer does, as the cfg.x.default_btag_variables
    # fallback of the tasks is not applied on the command line (cfg.x.default_variables wins)
    cf_law cf.BTagEfficiencyPlot --datasets "${CF_E2E_DATASET}" --dataset-group tt \
        --variables btag_jet_pt,btag_jet_eta || return "$?"
    local btag_eff
    btag_eff="$( find_one cf.BTagEfficiency btag_efficiency.pickle )" || return "$?"
    check_py check_btag_eff.py "${btag_eff}" || return "$?"
    assert_some cf.BTagEfficiencyPlot "*.pdf" || return "$?"
}
action "$@"
