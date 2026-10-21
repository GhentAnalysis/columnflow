#!/usr/bin/env bash

# Producers feature job of the test-template e2e workflow: single and combined producers, weighted
# and unweighted histograms, 1D/2D plots, yield tables and united columns, all on top of the reduced
# events of the base job. Run from the generated analysis directory.

source "$( dirname "${BASH_SOURCE[0]}" )/lib.sh" || exit "$?"

action() {
    local ds="--dataset ${CF_E2E_DATASET}"
    local dss="--datasets ${CF_E2E_DATASET}"
    local prods="--producers features,event_weights"

    # each producer on its own, and the combined default one
    cf_law cf.ProduceColumns ${ds} --producer features || return "$?"
    cf_law cf.ProduceColumns ${ds} --producer event_weights || return "$?"
    cf_law cf.ProduceColumns ${ds} --producer default || return "$?"

    # weighted histograms (all_weights, the default) from two producers, and unweighted ones
    # (cf_default) from a single one, which must differ in their sum of weights but not in entries
    cf_law cf.MergeHistograms ${ds} ${prods} --variables jet1_pt || return "$?"
    cf_law cf.MergeHistograms ${ds} --producers features --hist-producer cf_default --variables jet1_pt || return "$?"
    local h_weighted h_unweighted
    h_weighted="$( find_one cf.MergeHistograms hist__var_jet1_pt.pickle "*/hist__all_weights/*" )" || return "$?"
    h_unweighted="$( find_one cf.MergeHistograms hist__var_jet1_pt.pickle "*/hist__cf_default/*" )" || return "$?"
    check_py check_weighted_hists.py "${h_weighted}" "${h_unweighted}" || return "$?"

    # plots, including production-level categories and a 2D variable
    cf_law cf.PlotVariables1D ${dss} ${prods} --processes tt --variables jet1_pt,jet1_eta,n_jet,ht \
        --categories incl,2e,1e1mu,2mu || return "$?"
    assert_some cf.PlotVariables1D "*.pdf" || return "$?"
    cf_law cf.PlotVariables2D ${dss} ${prods} --processes tt --variables n_jet-jet1_pt --categories incl || return "$?"
    assert_some cf.PlotVariables2D "*.pdf" || return "$?"
    # TODO: add cf.PlotVariablesPerProcess2D once fixed; it inherits run() from PlotVariables2D while
    # its wrapper-style requires() is keyed by process, failing with "KeyError: <config>" in
    # get_config_process_map (columnflow/tasks/plotting.py)

    # yield table
    cf_law cf.CreateYieldTable ${dss} ${prods} --processes tt --categories incl,2e,1e1mu,2mu || return "$?"
    local yields
    yields="$( find_one cf.CreateYieldTable "yields__*.json" )" || return "$?"
    python3 "${CF_E2E_DIR}/check_yields.py" "${yields}" tt incl,2e,1e1mu,2mu || return "$?"

    # reduced events and produced columns united into a single file per branch
    cf_law cf.UniteColumns ${ds} ${prods} || return "$?"
    assert_some cf.UniteColumns "data_*.parquet" || return "$?"
}
action "$@"
