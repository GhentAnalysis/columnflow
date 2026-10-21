# Known issues found while building the test template

Bugs in `GhentAnalysis/master` (at `131cd8d9`) found while building the CI test template
(`tests/ci/test_template`, see `README.md`, section "Test template"). Each entry lists the location,
how to reproduce it and what happens. Entries are marked:

- **reproduced**: the failure was triggered by actually running the task on the CI test data,
- **code reading**: found by reading the code only, not triggered yet.

Where a bug blocks a task from the e2e scripts, the script carries a `TODO` pointing here; remove it
together with the entry once fixed.

## Tasks

### 1. `cf.PlotVariablesPerProcess2D` always fails (reproduced)

- **Location:** `columnflow/tasks/plotting.py`, `PlotVariablesPerProcess2D` (line 474) and
  `PlotVariablesBase.get_config_process_map` (line 108).
- **Reproduce:** `law run cf.PlotVariablesPerProcess2D --datasets tt_dl_powheg --processes tt
  --variables n_jet-jet1_pt --categories incl ...` on the test template.
- **Observed:** `KeyError: 'l18'` in `get_config_process_map`
  (`_req = reqs[config_inst.name][dataset_inst.name]`).
- **Cause:** the class derives from `PlotVariables2D` and `law.WrapperTask`. Its `requires()`
  returns wrapper-style requirements keyed by process (`{process: PlotVariables2D.req(...)}`), but
  `run()` and `output()` are still inherited from `PlotVariables2D`, so the task runs as a plot task
  whose requirements are not keyed by config and dataset. The same pattern probably affects the other
  `Per*` wrapper classes built like this.
- **Skipped in:** `tests/ci/e2e/producers.sh`.

### 2. `cf.PlotShiftedVariablesPerShift1D` always fails (reproduced)

- **Location:** `columnflow/tasks/plotting.py`, `PlotVariablesBaseMultiShifts.get_plot_shifts`
  (line 583), and `columnflow/plotting/plot_functions_1d.py`, `plot_shifted_variable` (line 283).
- **Reproduce:** `law run cf.PlotShiftedVariablesPerShift1D --datasets tt_dl_powheg --processes tt
  --variables jet1_pt --categories incl --shift-sources minbias_xs,jer ...`
- **Observed:** `KeyError: "'nominal' not in axis"` in `plot_shifted_variable`.
- **Cause:** with `combine_shifts = False`, each branch only gets the up and down shifts of its own
  source (`expand_shift_sources([self.branch_data.shift_source])`), but `plot_shifted_variable`
  always normalizes the ratio to the `nominal` bin. The per-shift branches must include nominal.
- **Skipped in:** `tests/ci/e2e/shifts.sh`.

### 3. `cf.MergeShiftedHistograms` silently writes an empty nominal (reproduced)

- **Location:** `columnflow/tasks/histograms.py`, `MergeShiftedHistograms.workflow_requires` /
  `requires` (around line 567).
- **Reproduce:** `law run cf.MergeShiftedHistograms --dataset tt_dl_powheg --variables jet1_pt
  --shift-sources minbias_xs,mur,muf ...` and inspect the output.
- **Observed:** the merged histogram has a `nominal` entry on its shift axis, but it is all zeros.
  The shifted entries are filled correctly.
- **Cause:** the comment says "add nominal and both directions per shift source", but nominal is
  only required when `nominal` is part of `--shift-sources` (`enforce_nominal_shift_source` is
  `False` for this task, `True` only for the plotting tasks). The zero `nominal` entry comes from the
  `shift` axis that the hist producer seeds with nominal (`columnflow/histogramming/default.py:54`),
  so nothing complains. Either always add nominal, or drop or flag the seeded empty bin.
- **Worked around in:** `tests/ci/e2e/shifts.sh`, by passing `nominal` explicitly.

### 4. `cf.BTagEfficiency` / `cf.BTagEfficiencyPlot` ignore `default_btag_variables` on the command line (reproduced)

- **Location:** `columnflow/tasks/cmsGhent/selection_hists.py`, `CustomDefaultVariablesMixin`
  (line 23).
- **Reproduce:** `law run cf.BTagEfficiencyPlot --datasets tt_dl_powheg --dataset-group tt ...`
  without `--variables`.
- **Observed:** the tasks are created with `variables=n_jet` (the config's `default_variables`), and
  `cf.BTagEfficiency` fails because the `btag_efficiencies` histogram has no `n_jet` axis.
- **Cause:** the mixin only replaces the variables with `cfg.x.default_btag_variables` when
  `params["variables"]` is empty, but by then it already holds the generic config default. The
  producer path (`fixed_wp_btag_weights`) is unaffected because it passes `variables` explicitly.
- **Worked around in:** `tests/ci/e2e/base.sh`, by passing `--variables btag_jet_pt,btag_jet_eta`.

### 5. `cf.BTagEfficiency` writes NaN efficiencies for empty bins (reproduced)

- **Location:** `columnflow/tasks/cmsGhent/btagefficiency.py`, `BTagEfficiency.run` (line 79), and
  `SelectionEfficiencyHistMixin.efficiency` (`selection_hists.py:178`).
- **Observed:** on the CI data, the 9 bins of charm jets with pT 400–1000 GeV and |eta| 1.5–2.5
  (3 working points × 3 systematics) are `0 / 0 = NaN`. They are written as such to the pickle and
  to the correctionlib JSON that `fixed_wp_btag_weights` evaluates.
- **Impact:** any jet falling into such a bin gets a NaN b-tag weight, which propagates into the
  event weight. On the CI data no selected jet falls into these bins, so nothing fails downstream.
  Empty bins should get a fallback (e.g. a neighbouring bin, the flavour average, or 0 / 1 with a
  warning).
- **Tolerated in:** `tests/ci/e2e/check_btag_eff.py` (warns, fails above 5% NaN bins).

### 6. Cutflow tasks cannot be run (reproduced)

- **Reproduce:** `law run cf.CreateCutflowHistograms --dataset tt_dl_powheg --variables cf_jet1_pt
  ...` or `law run cf.PlotCutflow --datasets tt_dl_powheg --processes tt --categories incl ...` on
  the test template.
- **Observed:** `AttributeError: 'Requirements' object has no attribute 'GetDatasetLFNs'` while the
  requirements are built.

- **Location:** `columnflow/tasks/cutflow.py`, `CreateCutflowHistograms` (lines 84-113 and
  `run()`).
- **Cause:**
  - `reqs` only declares `MergeSelectionMasks`, but `requires()` uses `self.reqs.GetDatasetLFNs`,
    `self.reqs.CalibrateEvents` and `self.reqs.SelectEvents`, so building the requirements raises
    `AttributeError`.
  - `requires()` also uses `self.norm_weight_producer_inst`, which only exists on
    `MergeSelectionMasks` (`selection.py:451`).
  - The `run()` body looks like a broken merge: an output key that does not exist is dumped before
    filling, and part of the logic sits outside the chunk loop.
  - In addition, `PlotCutflow` passes `missing_selector_step_strategy` to `MergeCutflowHistograms`,
    which has no such parameter.
- **Impact:** everything downstream is unusable: `MergeCutflowHistograms`, `PlotCutflow`,
  `PlotCutflowVariables1D/2D`, `PlotCutflowVariablesPerProcess2D`, cmsGhent `CreateCutflowTable` and
  `PlotInclusiveVariables1D`.
- **Related:** open PR #99 ("Fix of Cutflow CreateHistograms").

### 7. cmsGhent `CreateCutflowTable` selects histogram bins by id (code reading)

- **Location:** `columnflow/tasks/cmsGhent/cutflow.py`, lines 169 and 174.
- **Cause:** bins are selected with `hist.loc(p.id)` / `hist.loc(c.id)`, but the process and
  category axes of current histograms are string categories (names), so the lookup fails even once
  issue 6 is fixed.

### 8. cmsGhent plotting tasks use outdated APIs (reproduced for `PlotVariables1DCatsPerProcess`)

- **Reproduce:** `law run cf.PlotVariables1DCatsPerProcess --datasets tt_dl_powheg --processes tt
  --variables jet1_pt --categories incl,2e,1e1mu,2mu ...`
- **Observed:** `ValueError: unknown process: ('tt',)` at `columnflow/tasks/cmsGhent/plotting.py:58`
  (`self.config_inst.get_process(self.branch_data.process)`). The branch data still holds the
  per-config parameter tuple of the multi-config task interface rather than a process name. The other
  problems below are hit once this one is fixed.

- **Location:** `columnflow/tasks/cmsGhent/plotting.py` (`PlotVariables1DCatsPerProcess`,
  `PlotVariables2DMigration`, `PlotVariables1DMultiVar`, `PlotROC`).
- **Cause:**
  - They iterate `self.input().items()` as if keyed by dataset (lines 65, 200), but the inputs are
    now keyed by config, then dataset.
  - They select histogram bins by `p.id` / `c.id` / `s.id`, but the axes now use names.
  - They call `self.invoke_hist_hook(hists)` (line 239) and read `self.hist_hook` (line 354), but
    the mixin only provides `invoke_hist_hooks` / `hist_hooks` (`framework/mixins.py:2805, 2827`).

### 9. cmsGhent `SelectionHistPlot` cannot be created (reproduced)

- **Reproduce:** `law run cf.SelectionHistPlot --datasets tt_dl_powheg --hist-name btag_efficiencies
  --variables btag_jet_pt,btag_jet_eta ...`
- **Observed:** `KeyError: 'calibrators'` during parameter resolution, in
  `CalibratorClassesMixin.resolve_instances` (`columnflow/tasks/framework/mixins.py:383`). The task
  does not get the `calibrators` parameter the mixin expects, so it cannot be instantiated at all.
- **Further, once that is fixed (code reading):** `columnflow/tasks/cmsGhent/selection_hists.py`
  lines 272 and 277 use `self.control_plot_function` and `self.get_control_plot_parameter()`, which
  are not defined anywhere in columnflow.

### 10. `cf.PlotInclusiveVariables1D` is never registered (code reading)

- **Location:** `columnflow/tasks/cmsGhent/plotting2.py`.
- **Cause:** no module imports `plotting2.py` and it is not listed in `law.cfg [modules]`, so
  `law index` never finds the task.

### 11. `[outputs]` key `cf.CreatePileupWeights` never matches (code reading)

- **Location:** `analysis_templates/ghent_template/law.cfg`, line 134.
- **Cause:** the task lives in the `cf.cms` namespace (`columnflow/tasks/cms/external.py:28`), so its
  family is `cf.cms.CreatePileupWeights` and the `cf.CreatePileupWeights` key has no effect.

## ghent_template TAFs and configuration

These live in `analysis_templates/ghent_template/__cf_module_name__/`. The test template fixes or
removes them in its copy, but the ghent_template itself still ships them to every new analysis.

### 12. `inference/example.py` cannot be used with this config (code reading)

- **Cause:** the model was ported from cms_minimal and refers to things the ghent config does not
  define: variable `jet1_eta` (fails first, in `columnflow/tasks/framework/inference.py`), category
  `2j`, process `st`, datasets `st_tchannel_t_powheg` and `data_mu_b`, and shift sources `tune`, `mu`
  and `jec` (the JEC shifts are called `jec_<source>`). It also uses `tt_sl_powheg`, which has no file
  in the CI bundle.
- **Fixed in test template:** replaced by `inference/test.py`. Open PR #118 also reworks this model.

### 13. `histogramming/example.py` raises on MC (code reading)

- **Location:** `example_init`.
- **Cause:** `self.weight_columns = {}` creates a dict, so the following
  `self.weight_columns |= {"normalization_weight", "muon_weight"}` raises `TypeError`. It also
  declares the shift source `mu`, which this config does not define.
- **Fixed in test template:** removed.

### 14. `ml/example.py` cannot run (code reading)

- **Cause:** it declares `produces` as `"{cls_name}.ouptut"` (typo, line 54), while the variable and
  `evaluate` use `.output`. It trains on `st_tchannel_t_powheg`, which is not in the config, and has
  no `training_calibrators/selectors/producers`.
- **Fixed in test template:** removed (not registered in `ml_modules`).

### 15. The default hist producer applies no event weights, and `all_weights` would count the pileup weight twice (code reading)

- **Location:** `config/config___cf_short_name_lc__.py` (`default_hist_producer = "cf_default"`),
  `config/shifts.py` (the `event_weights` loop).
- **Cause:**
  - `cf_default` fills every event with weight 1, so all ghent_template histograms and plots are
    unweighted (no normalization, no scale factors). The note in `shifts.py` says `all_weights` was
    the intended default.
  - Switching to `all_weights` as is would apply the pileup weight twice: the `minbias_xs` shift
    aliases both `pu_weight` and `normalized_pu_weight`, so both end up in `cfg.x.event_weights`, and
    `all_weights` multiplies every entry.
- **Fixed in test template:** `all_weights` as default, `pu_weight` removed from
  `cfg.x.event_weights` (`config/shifts.py`).

### 16. No TAF declares the `jer` / `jec_*` / `top_pt` shifts (code reading)

- **Location:** `calibration/default.py`, `selection/default.py`.
- **Cause:** the config defines these selection dependent shifts with column aliases, and the
  calibrator produces the varied columns, but neither the calibrator nor the selector adds them to
  `self.shifts`. `--shift jer_up` therefore silently falls back to nominal, and
  `--shift-sources jer` is dropped. `top_pt` aliases `top_pt_weight`, which nothing produces.
- **Fixed in test template:** the `default` selector declares the `jer` shifts. `jec_*` and `top_pt`
  remain undeclared.

### 17. Lepton channel categories do not cover the inclusive category (reproduced)

- **Location:** `selection/categories.py`, `catid_2e` / `catid_1e1mu` / `catid_2mu` (line 55 ff).
- **Observed:** on the CI data, the `2e`, `1e1mu` and `2mu` yields add up to 98.7% of the `incl`
  yield.
- **Cause:** the production-level categorizers count all leptons left in the reduced
  `Electron`/`Muon` collections, while the selection requires exactly two leptons passing its cuts.
  Events with an additional lepton in the reduced collections fall into no channel. Whether that is
  intended is an analysis decision, but it differs from the selection-level `catid_selection_*`
  categorizers, which count the selected objects.
- **Tolerated in:** `tests/ci/e2e/check_yields.py` (requires ≥ 90% coverage).

### 18. The `VetoTrigger` selection step is always False (code reading)

- **Location:** `selection/trigger.py`, lines 65-80.
- **Cause:** `veto_trigger` starts as an all-False mask and is only ever AND-ed, so it stays False.
  The event selection itself is correct (the veto is applied to `sel_trigger`), but stats and
  cutflows show the `VetoTrigger` step as removing every event.

### 19. Python imports use `__cf_short_name_lc__` while the package is `__cf_module_name__` (code reading)

- **Location:** 8 modules, e.g. `config/config___cf_short_name_lc__.py`, `production/default.py`.
- **Cause:** the package directory and `law.cfg` use the module name placeholder, but the imports
  use the short name placeholder. A generated analysis only imports correctly when the module name
  equals the lower-case short name, which `create_analysis.sh` defaults to but does not enforce.
