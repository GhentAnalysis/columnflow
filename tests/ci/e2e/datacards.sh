#!/usr/bin/env bash

# Datacards feature job of the test-template e2e workflow: writes the datacard of the "test"
# inference model (tests/ci/test_template/__cf_module_name__/inference/test.py), whose shape
# parameters require shifted histograms of event weight shifts (minbias_xs, mur) and of the selection
# dependent jer shift. Run from the generated analysis directory.

source "$( dirname "${BASH_SOURCE[0]}" )/lib.sh" || exit "$?"

action() {
    cf_law cf.CreateDatacards --inference-model test || return "$?"

    local card shapes
    card="$( find_one cf.CreateDatacards datacard__incl.txt )" || return "$?"
    shapes="$( find_one cf.CreateDatacards shapes__incl.root )" || return "$?"
    check_py check_datacard.py "${card}" "${shapes}" incl TT minbias_xs,mur,jer || return "$?"
}
action "$@"
