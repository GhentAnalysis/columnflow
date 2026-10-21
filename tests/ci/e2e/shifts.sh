#!/usr/bin/env bash

# Shifts feature job of the test-template e2e workflow: event weight shifts (only histogramming runs
# per shift) and the selection dependent jer shift (declared by the default selector, so selection,
# reduction, production and histogramming re-run at jer_up/down), merged into shifted histograms and
# plotted. Run from the generated analysis directory.

source "$( dirname "${BASH_SOURCE[0]}" )/lib.sh" || exit "$?"

action() {
    local ds="--dataset ${CF_E2E_DATASET}"
    local dss="--datasets ${CF_E2E_DATASET}"

    # config-level (minbias_xs) and dataset-level (mur, muf) event weight shifts; nominal must be
    # requested explicitly, otherwise the merged histogram only has an empty nominal bin (seeded by
    # the hist producer) as MergeShiftedHistograms does not add it (see tests/ci/known_issues.md)
    cf_law cf.MergeShiftedHistograms ${ds} --variables jet1_pt --shift-sources nominal,minbias_xs,mur,muf || return "$?"
    # (store directories are named after the number of sources, the first sources and a hash of all)
    local h_weights
    h_weights="$( find_one cf.MergeShiftedHistograms hists__jet1_pt.pickle "*/shifts__4__nominal__minbias_xs__muf__*" )" || return "$?"
    check_py check_shift_hists.py "${h_weights}" minbias_xs,mur,muf || return "$?"

    # selection dependent shift
    cf_law cf.MergeShiftedHistograms ${ds} --variables jet1_pt --shift-sources nominal,jer || return "$?"
    local h_jer
    h_jer="$( find_one cf.MergeShiftedHistograms hists__jet1_pt.pickle "*/shifts__2__nominal__jer/*" )" || return "$?"
    check_py check_shift_hists.py "${h_jer}" jer || return "$?"

    # the jer shifts must have re-run the selection and reduction rather than silently falling back
    # to nominal, which would also make the check above fail as up == nominal
    local shift
    for shift in jer_up jer_down; do
        assert_some cf.MergeSelectionStats stats.json "*/${shift}/*" || return "$?"
        assert_some cf.ReduceEvents "*.parquet" "*/${shift}/*" || return "$?"
    done

    # shifted plots, combining weight and selection dependent sources
    cf_law cf.PlotShiftedVariables1D ${dss} --processes tt --variables jet1_pt --categories incl \
        --shift-sources minbias_xs,jer || return "$?"
    assert_some cf.PlotShiftedVariables1D "*.pdf" || return "$?"
    # TODO: add cf.PlotShiftedVariablesPerShift1D once fixed; its per-shift branches do not pass the
    # nominal shift to plot_shifted_variable, failing with "KeyError: 'nominal' not in axis"
    # (see tests/ci/known_issues.md)
}
action "$@"
