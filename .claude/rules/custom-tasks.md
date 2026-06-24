---
paths:
  - "**/tasks/**"
---

# Custom Law Tasks

Custom tasks extend the columnflow pipeline for non-standard outputs (custom plots, derived datacards, scale factors, etc.).

## Skeleton

```python
import law
from columnflow.tasks.framework.base import AnalysisTask, ConfigTask, Requirements

class MyTask(ConfigTask, law.LocalWorkflow):
    # class-level task_namespace for the analysis (set once in the analysis base class)
    task_namespace = "hbw"

    # Upstream task requirements
    def requires(self):
        return SomeUpstreamTask.req(self)

    # Branch map: maps branch index → payload (for LocalWorkflow only)
    def create_branch_map(self):
        return list(self.categories)          # one branch per category

    # Output target(s) for this branch
    def output(self):
        return {
            "data": self.local_target("result.json"),
            "plot": self.local_target("plot.pdf"),
        }

    # Actual computation
    def run(self):
        inp = self.input()
        data = inp["data"].load(formatter="json")
        # ... compute ...
        self.output()["data"].dump(result, formatter="json")
```

## Analysis base class (define once per analysis)

```python
class HBWTask(AnalysisTask):
    task_namespace = "hbw"          # prefix for all custom tasks
```

## Pattern 1 — Histogram-consuming task (trigger SF, custom plots from histograms)

Inherits `HistogramsUserSingleShiftBase` which already wires up `MergeHistograms` requirements for all configured datasets and hist_producers.

```python
from columnflow.tasks.framework.histograms import HistogramsUserSingleShiftBase
from columnflow.tasks.framework.mixins import DatasetsProcessesMixin

class ComputeMyScaleFactors(HBWTask, HistogramsUserSingleShiftBase, DatasetsProcessesMixin, law.LocalWorkflow):

    def create_branch_map(self):
        return list(self.hist_producer_insts)  # one branch per hist_producer

    def requires(self):
        # HistogramsUserSingleShiftBase provides .requires_histograms() helper
        reqs = {}
        for dataset_inst in self.dataset_insts:
            reqs[dataset_inst.name] = MergeHistograms.req(
                self,
                dataset=dataset_inst.name,
                branch=-1,
            )
        return reqs

    def output(self):
        return {
            "json": self.local_target("corrections.json"),
            "plot": self.local_target("efficiency.pdf"),
        }

    def run(self):
        inputs = self.input()
        # load histograms using inherited helper
        hist = self.load_histogram(inputs, self.config_inst, dataset_inst, variable)
        # compute data/MC ratios, write CorrectionLib JSON, save plots
        self.output()["json"].dump(corrections, formatter="json")
```

## Pattern 2 — Custom datacard writer (per-category LocalWorkflow)

Inherits `SerializeInferenceModelBase` which handles inference model + histogram loading.

```python
from columnflow.tasks.framework.inference import SerializeInferenceModelBase
from columnflow.tasks.histograms import MergeHistograms

class CreateMultipleDatacards(HBWTask, SerializeInferenceModelBase, law.LocalWorkflow):

    def create_branch_map(self):
        # one branch per analysis category
        return {i: cat for i, cat in enumerate(self.inference_model_inst.categories)}

    def requires(self):
        cat = self.branch_data
        reqs = {}
        for dataset_inst in self.dataset_insts:
            reqs[dataset_inst.name] = MergeHistograms.req(
                self,
                dataset=dataset_inst.name,
                branch=-1,
            )
        return reqs

    def output(self):
        cat = self.branch_data
        return {
            "datacard": self.local_target(f"{cat.name}/datacard.txt"),
            "shapes":   self.local_target(f"{cat.name}/shapes.root"),
        }

    def run(self):
        inputs = self.input()
        cat = self.branch_data
        # write datacard for this category
        writer = DatacardWriter(self.inference_model_inst, cat)
        writer.run(inputs, self.output())
```

## Pattern 3 — Custom plot task (reads external ROOT file)

```python
from columnflow.plots.util import PlotBase1D
from columnflow.tasks.framework.mixins import (
    DatasetsProcessesMixin, CalibratorClassesMixin, SelectorClassMixin,
    ProducerClassesMixin, HistProducerClassMixin, InferenceModelMixin,
)

class PlotPostfitShapes(
    HBWTask,
    PlotBase1D,
    DatasetsProcessesMixin,
    InferenceModelMixin,
    HistProducerClassMixin,
):
    # No upstream law tasks required — reads an external file directly
    def requires(self):
        return {}

    def output(self):
        return law.LocalDirectoryTarget(self.local_path("plots"))

    def run(self):
        fit_file = self.input()  # or open an external file
        # load histograms from ROOT, call self.call_plot_func(...)
        self.output().touch()
```

## Registering custom tasks in law.cfg

```ini
[modules]
# ... existing columnflow modules ...
hbw.tasks.trigger_sf
hbw.tasks.multiple_datacards
hbw.tasks.postfit_plots
```

Run with:
```bash
law run hbw.ComputeMyScaleFactors --version dev1 --config run2_2018 --branch 0
law run hbw.CreateMultipleDatacards --version dev1 --inference-model default
```

## Key rules

- Always call `.req(self, ...)` on upstream tasks instead of constructing them directly — this propagates shared parameters.
- `self.local_target(path)` creates a `law.LocalFileTarget` inside the task's store path. For directories use `law.LocalDirectoryTarget(self.local_path(...))`.
- `law.LocalWorkflow` requires `create_branch_map()` returning a dict or list; each entry becomes one branch.
- Without `LocalWorkflow`, the task is a single-branch task — omit `create_branch_map()`.
- For remote execution add `HTCondorWorkflow` or `SlurmWorkflow` as additional base classes.
- Always register the module path in `law.cfg [modules]` so `law` can discover the task.
