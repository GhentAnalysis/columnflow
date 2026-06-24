# 06 — Writing Custom Analysis Code

This guide shows how to write the five types of Task Array Functions (TAFs) and custom law tasks for your analysis. Examples are inspired by the [hh2bbww analysis](https://github.com/uhh-cms/hh2bbww) and the Ghent analysis template.

---

## Object Selection Selector

The most common starting point: a set of internal selectors that define the baseline objects (muons, electrons, jets), then one exposed selector that composes them.

### Step 1: Object-level Selectors (unexposed, internal)

```python
# selection/objects.py
from collections import defaultdict
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column, sorted_indices_from_mask
from columnflow.util import maybe_import, four_vec

ak = maybe_import("awkward")


@selector(
    uses=four_vec("Muon", {"sip3d", "dxy", "dz", "miniPFRelIso_all", "mediumId"}),
)
def muon_object(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    # Loose muon selection
    mu_mask = (
        (abs(events.Muon.eta) < 2.4) &
        (events.Muon.pt > 10.0) &
        (events.Muon.miniPFRelIso_all < 0.4) &
        (events.Muon.sip3d < 8) &
        (abs(events.Muon.dxy) < 0.05) &
        (abs(events.Muon.dz) < 0.1)
    )
    # Store tight flag as a new column for later use
    mu_tight = mu_mask & events.Muon.mediumId
    events = set_ak_column(events, "Muon.tight", mu_tight, value_type=bool)

    return events, SelectionResult(
        objects={"Muon": {"Muon": sorted_indices_from_mask(mu_mask, events.Muon.pt)}},
    )


@selector(
    uses=four_vec("Jet", {"btagDeepFlavB"}),
)
def jet_object(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    jet_mask = (
        (events.Jet.pt > 25.0) &
        (abs(events.Jet.eta) < 2.4)
    )
    jet_indices = ak.local_index(events.Jet.pt, axis=1)[jet_mask]

    return events, SelectionResult(
        objects={"Jet": {"Jet": jet_indices}},
    )
```

### Step 2: Event-level Selectors (unexposed, internal)

```python
# selection/event_selection.py
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import

ak = maybe_import("awkward")


@selector(
    uses=four_vec("Muon"),
)
def lepton_selection(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # work with the already-selected muons (object indices from object_selection step)
    muon = events.Muon[results.objects.Muon.Muon]
    n_mu = ak.sum(muon.pt > 0, axis=1)

    return events, SelectionResult(steps={"lepton": n_mu >= 1})


@selector(
    uses=four_vec("Jet"),
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    jet = events.Jet[results.objects.Jet.Jet]
    n_jet = ak.sum(jet.pt > 0, axis=1)
    n_bjet = ak.sum(jet.btagDeepFlavB > 0.5, axis=1)  # medium WP

    return events, SelectionResult(
        steps={
            "jet": n_jet >= 2,
            "btag": n_bjet >= 1,
        },
    )
```

### Step 3: The Exposed (Top-Level) Selector

```python
# selection/default.py
from collections import defaultdict
from operator import and_
from functools import reduce

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.processes import process_ids
from columnflow.production.util import attach_coffea_behavior
from columnflow.selection.stats import increment_stats
from columnflow.util import maybe_import

from myanalysis.selection.objects import muon_object, jet_object
from myanalysis.selection.event_selection import lepton_selection, jet_selection

ak = maybe_import("awkward")


@selector(
    uses={
        attach_coffea_behavior, mc_weight, process_ids,
        muon_object, jet_object,
        lepton_selection, jet_selection,
        increment_stats,
    },
    produces={
        attach_coffea_behavior, mc_weight, process_ids,
    },
    exposed=True,   # reachable from CLI as --selector default
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    # Attach coffea 4-vector behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # MC bookkeeping
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
    events = self[process_ids](events, **kwargs)

    results = SelectionResult()

    # Object selection
    events, muon_results = self[muon_object](events, stats, **kwargs)
    results += muon_results

    events, jet_results = self[jet_object](events, results, stats, **kwargs)
    results += jet_results

    # Event-level selection
    events, lep_results = self[lepton_selection](events, results, **kwargs)
    results += lep_results

    events, jet_ev_results = self[jet_selection](events, results, **kwargs)
    results += jet_ev_results

    # Combine all selection steps into the final event mask
    results.event = reduce(and_, results.steps.values())

    # Increment selection statistics (saved in stats.json)
    events, results = self[increment_stats](
        events,
        results,
        stats,
        weight_map={
            "mc_weight": (events.mc_weight if self.dataset_inst.is_mc else None, Ellipsis),
            "mc_weight_selected": (events.mc_weight if self.dataset_inst.is_mc else None, results.event),
        },
        **kwargs,
    )

    return events, results
```

---

## Producer: High-Level Variables

```python
# production/features.py
from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.util import maybe_import, four_vec

ak = maybe_import("awkward")
np = maybe_import("numpy")


@producer(
    uses=four_vec("Jet") | four_vec("Electron") | four_vec("Muon"),
    produces={"ht", "n_jet", "n_bjet", "n_electron", "n_muon", "jet1_pt", "lep1_pt"},
)
def kinematic_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # Scalar sum of jet pT
    events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1), value_type=np.float32)

    # Object counts
    events = set_ak_column(events, "n_jet", ak.num(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_bjet", ak.sum(events.Jet.btagDeepFlavB > 0.5, axis=1))
    events = set_ak_column(events, "n_electron", ak.num(events.Electron.pt, axis=1))
    events = set_ak_column(events, "n_muon", ak.num(events.Muon.pt, axis=1))

    # Leading jet pT (EMPTY_FLOAT if no jets)
    from columnflow.columnar_util import Route
    events = set_ak_column(
        events, "jet1_pt",
        Route("Jet.pt[:,0]").apply(events, null_value=EMPTY_FLOAT),
        value_type=np.float32,
    )

    # Leading lepton pT (combine muons + electrons)
    lep_pt = ak.concatenate([events.Muon.pt, events.Electron.pt], axis=1)
    lep_pt = lep_pt[ak.argsort(lep_pt, axis=1, ascending=False)]
    events = set_ak_column(
        events, "lep1_pt",
        ak.fill_none(ak.pad_none(lep_pt, 1, axis=1)[:, 0], EMPTY_FLOAT),
        value_type=np.float32,
    )

    return events
```

---

## Producer: Event Weights

```python
# production/weights.py
from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import
from columnflow.config_util import get_shifts_from_sources

ak = maybe_import("awkward")
np = maybe_import("numpy")


@producer(
    uses={
        "normalization_weight",  # added by columnflow's normalization_weights producer
        "muon_weight",           # added by columnflow's muon_weights producer
        "pu_weight",             # added by columnflow's pileup_weights producer
    },
    produces={"event_weight"},
)
def event_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    weight = ak.Array(np.ones(len(events), dtype=np.float32))

    if self.dataset_inst.is_mc:
        weight = weight * events.normalization_weight
        weight = weight * events.muon_weight
        weight = weight * events.pu_weight

    events = set_ak_column(events, "event_weight", weight, value_type=np.float32)
    return events


@event_weights.init
def event_weights_init(self: Producer) -> None:
    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return
    # declare that this producer responds to muon and pileup shift sources
    self.shifts |= set(get_shifts_from_sources(self.config_inst, "mu", "minbias_xs"))
```

---

## Calibrator: Custom Jet Energy Correction

```python
# calibration/jets.py
from columnflow.calibration import Calibrator, calibrator
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

ak = maybe_import("awkward")
np = maybe_import("numpy")


@calibrator(
    uses={"Jet.{pt,eta,phi,mass,rawFactor,area}", "fixedGridRhoFastjetAll"},
    produces={"Jet.{pt,eta,phi,mass}"},   # overwrites nominal; varied columns added in init
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    # Placeholder: in practice, load JEC from external files in setup() hook
    # and apply them here per-event on the Jet collection.
    corrected_pt = events.Jet.pt * self.jec_factor  # self.jec_factor set in setup()
    events = set_ak_column(events, "Jet.pt", corrected_pt)
    return events


@default.requires
def default_requires(self: Calibrator, task, reqs: dict) -> None:
    from columnflow.tasks.external import BundleExternalFiles
    reqs["ext"] = BundleExternalFiles.req(task)


@default.setup
def default_setup(self: Calibrator, task, reqs, inputs, reader_targets) -> None:
    # Load JEC files from the bundled external files
    bundle = inputs["ext"]["collection"][0]
    # ... parse JEC file from bundle["jec"] ...
    self.jec_factor = 1.02  # placeholder


@default.init
def default_init(self: Calibrator) -> None:
    from columnflow.config_util import get_shifts_from_sources
    # declare which shifts this calibrator covers
    self.shifts |= set(get_shifts_from_sources(self.config_inst, "jec"))
    # for each shift, add the varied column to produces
    for shift_inst in self.shifts:
        if "jec" in shift_inst.name:
            self.produces.add(f"Jet.pt_{shift_inst.name}")
```

---

## Categorizer

Categorizers define analysis regions used in histogramming and plotting.

```python
# categorization/categories.py
from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import

ak = maybe_import("awkward")


@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ak.ones_like(events.event, dtype=bool)


@categorizer(uses={"n_jet"})
def cat_2j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.n_jet >= 2


@categorizer(uses={"n_jet", "n_bjet"})
def cat_sr(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, (events.n_jet >= 4) & (events.n_bjet >= 2)


@categorizer(uses={"n_jet", "n_bjet"})
def cat_cr_1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, (events.n_jet >= 4) & (events.n_bjet == 1)
```

Register them in the config:

```python
from columnflow.config_util import add_category
add_category(cfg, name="incl",  id=1, selection="cat_incl",  label="Inclusive")
add_category(cfg, name="sr",    id=2, selection="cat_sr",    label="Signal Region")
add_category(cfg, name="cr_1b", id=3, selection="cat_cr_1b", label="CR (1b)")
```

And register the file in `law.cfg`:

```ini
categorization_modules: myanalysis.categorization.categories
```

---

## HistProducer

The HistProducer controls event weighting and the histogram filling loop. The most common pattern extends columnflow's `cf_default`:

```python
# histogramming/default.py
from columnflow.histogramming import HistProducer
from columnflow.histogramming.default import cf_default
from columnflow.columnar_util import Route
from columnflow.util import maybe_import
from columnflow.config_util import get_shifts_from_sources

ak = maybe_import("awkward")
np = maybe_import("numpy")


@cf_default.hist_producer()
def default(self: HistProducer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    weight = ak.Array(np.ones(len(events), dtype=np.float32))

    if self.dataset_inst.is_mc:
        for column in self.weight_columns:
            weight = weight * Route(column).apply(events)

    return events, weight


@default.init
def default_init(self: HistProducer) -> None:
    self.weight_columns = set()
    if self.dataset_inst.is_data:
        return

    self.weight_columns |= {"normalization_weight", "muon_weight", "pu_weight"}
    self.uses |= self.weight_columns

    self.shifts |= set(get_shifts_from_sources(self.config_inst, "mu", "minbias_xs"))
```

---

## InferenceModel (Datacards)

```python
# inference/default.py
from columnflow.inference import inference_model, ParameterType, ParameterTransformation


@inference_model
def default(self):

    # --- Categories ---
    self.add_category(
        "sr",
        config_category="sr",
        config_variable="jet1_pt",
        config_data_datasets=["data_mu_b", "data_mu_c", "data_mu_d"],
        mc_stats=True,
    )

    # --- Processes ---
    self.add_process("TT", is_signal=False, config_process="tt")
    self.add_process("ST", is_signal=True,  config_process="st")
    self.add_process("WJets", is_signal=False, config_process="wjets")

    # --- Parameters ---
    # Luminosity (rate uncertainty)
    lumi = self.config_inst.x.luminosity
    for unc_name in lumi.uncertainties:
        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
            transformations=[ParameterTransformation.symmetrize],
        )

    # Muon scale factor (weight-based shape uncertainty)
    self.add_parameter(
        "mu",
        process=["TT", "ST", "WJets"],
        type=ParameterType.shape,
        config_shift_source="mu",
    )

    # JEC (selection-modifying shape uncertainty)
    self.add_parameter(
        "jec",
        process=["TT", "ST", "WJets"],
        type=ParameterType.shape,
        config_shift_source="jec",
    )

    # Tune (dedicated dataset)
    self.add_parameter(
        "tune",
        process="TT",
        type=ParameterType.shape,
        config_shift_source="tune",
    )
```

---

## Custom Law Task

For analysis steps that do not fit the standard TAF pattern, you can write a custom law task. The base class `BaseTask` from columnflow provides access to analysis/config/campaign objects.

```python
# tasks/base.py
from columnflow.tasks.framework.base import BaseTask


class MyAnalysisTask(BaseTask):
    task_namespace = "my"


# tasks/yield_table.py
import law
import order as od

from columnflow.tasks.framework.base import AnalysisTask
from columnflow.tasks.histograms import MergeHistograms


class CreateYieldTable(AnalysisTask, law.LocalWorkflow):
    """
    Custom task that reads merged histograms and writes a LaTeX yield table.
    """

    version = luigi.Parameter()
    processes = law.CSVParameter(default=("tt", "wjets"))

    def requires(self):
        return {
            proc: MergeHistograms.req(
                self,
                datasets=self.config_inst.get_process(proc).datasets,
                variables=("n_jet",),
            )
            for proc in self.processes
        }

    def output(self):
        return self.local_target("yield_table.tex")

    def run(self):
        import hist

        rows = []
        for proc, inp in self.input().items():
            h = inp.load()
            n_events = h.sum().value
            rows.append((proc, f"{n_events:.1f}"))

        table = "\\begin{tabular}{lr}\n"
        for proc, count in rows:
            table += f"  {proc} & {count} \\\\\n"
        table += "\\end{tabular}\n"

        self.output().dump(table, formatter="text")
```

Register in `law.cfg`:

```ini
[modules]
myanalysis.tasks
```

Run it:

```bash
law run my.CreateYieldTable --version dev1 --processes tt,wjets
```

---

## Composing the Main Producer

The main `default` producer (called from `--producers default`) should call sub-producers and set category IDs:

```python
# production/default.py
from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.util import maybe_import

from myanalysis.production.features import kinematic_features
from myanalysis.production.weights import event_weights

ak = maybe_import("awkward")


@producer(
    uses={category_ids, kinematic_features, normalization_weights, event_weights},
    produces={category_ids, kinematic_features, normalization_weights, event_weights},
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Reproduce category IDs (needed for histogramming)
    events = self[category_ids](events, **kwargs)

    # High-level kinematic features
    events = self[kinematic_features](events, **kwargs)

    # Normalization weight (xsec × lumi / sum_mc_weight)
    if self.dataset_inst.is_mc:
        events = self[normalization_weights](events, **kwargs)
        events = self[event_weights](events, **kwargs)

    return events


@default.init
def default_init(self: Producer) -> None:
    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return
    # MC-only sub-producers declared dynamically
    self.uses.add(event_weights)
    self.produces.add(event_weights)
```

---

## Common Patterns from hh2bbww

These patterns appear throughout the hh2bbww analysis and are good models to follow:

### Splitting config setup into helper functions

```python
# config/config_2018.py
def add_processes_and_datasets(cfg, campaign):
    from myanalysis.config import processes as procs
    cfg.add_process(procs.tt)
    cfg.add_process(procs.wjets)
    cfg.add_dataset(campaign.get_dataset("tt_dl_powheg"))
    cfg.add_dataset(campaign.get_dataset("wjets_madgraph"))

def add_variables(cfg):
    cfg.add_variable(name="jet1_pt", expression="Jet.pt[:,0]", ...)
    cfg.add_variable(name="n_jet",   expression="n_jet", ...)

def add_categories(cfg):
    from myanalysis.categorization.categories import cat_incl, cat_sr
    add_category(cfg, name="incl", id=1, selection="cat_incl", label="Inclusive")
    add_category(cfg, name="sr",   id=2, selection="cat_sr",   label="SR")

def create_config(analysis, campaign, config_name, config_id):
    cfg = analysis.add_config(campaign, name=config_name, id=config_id)
    add_processes_and_datasets(cfg, campaign)
    add_variables(cfg)
    add_categories(cfg)
    return cfg
```

### Using `four_vec` to declare vector columns compactly

```python
from columnflow.util import four_vec

@selector(
    uses=four_vec({"Jet", "Electron", "Muon"}, {"btagDeepFlavB"}) | {"MET.pt", "MET.phi"},
)
def my_selector(self, events, **kwargs): ...
```

`four_vec(collections, extra_fields)` expands to `Collection.{pt,eta,phi,mass,*extra_fields}` for each collection.

### Dynamic `uses`/`produces` in `init` hooks

```python
@my_producer.init
def my_producer_init(self: Producer) -> None:
    # Only add MC-specific columns for MC datasets
    if not getattr(self, "dataset_inst", None):
        return
    if self.dataset_inst.is_mc:
        self.uses.add("GenJet.pt")
        self.produces.add("n_gen_jet")
```

### Composing SelectionResults

```python
results = SelectionResult()
events, r1 = self[muon_selection](events, stats, **kwargs)
results += r1
events, r2 = self[jet_selection](events, results, stats, **kwargs)
results += r2
# Combined event mask
results.event = reduce(and_, results.steps.values())
```

### Accessing config objects inside a TAF

```python
# self.config_inst   → order.Config
# self.dataset_inst  → order.Dataset
# self.analysis_inst → order.Analysis
# self.campaign      → order.Campaign (via self.config_inst.campaign)

year = self.config_inst.campaign.x.year
btag_wp = self.config_inst.x.btag_working_points["deepjet"]["medium"]
is_mc   = self.dataset_inst.is_mc
```
