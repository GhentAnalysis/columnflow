from __future__ import annotations

import law
import luigi

import order as od
from collections import OrderedDict

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    CalibratorClassesMixin, VariablesMixin, SelectorClassMixin, DatasetsMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase2D,
)
from columnflow.tasks.cmsGhent.selection_hists import SelectionEfficiencyHistMixin, CustomDefaultVariablesMixin

from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.util import dev_sandbox, dict_add_strict, DotDict, maybe_import

hist = maybe_import("hist")


class BTagEfficiencyBase:
    tag_name = "btag"
    flav_name = "hadronFlavour"
    flavours = {0: "light", 4: "charm", 5: "bottom"}
    wps = ["L", "M", "T"]

    single_config = True


class BTagEfficiency(
    BTagEfficiencyBase,
    SelectionEfficiencyHistMixin,
    CustomDefaultVariablesMixin,
    VariablesMixin,
    SelectorClassMixin,
    CalibratorClassesMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    def output(self):
        return {
            "json": self.target(f"{self.tag_name}_efficiency.json"),
            "hist": self.target(f"{self.tag_name}_efficiency.pickle"),
        }

    def get_plot_parameters(self):
        # convert parameters to usable values during plotting
        params = super().get_plot_parameters()
        dict_add_strict(params, "legend_title", "Processes")
        return params

    @law.decorator.log
    def run(self):
        import hist
        import correctionlib
        import correctionlib.convert
        from columnflow.plotting.cmsGhent.plot_util import cumulate

        variable_insts = list(map(self.config_inst.get_variable, self.variables))
        histograms = self.read_hist(variable_insts)
        sum_histogram = sum(histograms.values())

        # combine tagged and inclusive histograms to an efficiency histogram
        cum_histogram = cumulate(sum_histogram, direction="above", axis=f"{self.tag_name}_wp")
        incl = cum_histogram[{f"{self.tag_name}_wp": slice(0, 1)}]

        axes = OrderedDict(zip(cum_histogram.axes.name, cum_histogram.axes))
        axes[f"{self.tag_name}_wp"] = hist.axis.StrCategory(self.wps, name=f"{self.tag_name}_wp", label="working point")

        selected_counts = hist.Hist(*axes.values(), name=sum_histogram.name, storage=hist.storage.Weight())
        selected_counts.view()[:] = cum_histogram[{"btag_wp": slice(1, None)}].view()

        efficiency_hist = self.efficiency(selected_counts, incl)

        # save as pickle hist
        self.output()["hist"].dump(efficiency_hist, formatter="pickle")

        # save as correctionlib file
        efficiency_hist.label = "out"
        description = f"{self.tag_name} efficiencies of jets for {efficiency_hist.name} algorithm"
        clibcorr = correctionlib.convert.from_histogram(efficiency_hist[{"systematic": "central"}])
        clibcorr.description = description

        cset = correctionlib.schemav2.CorrectionSet(schema_version=2, description=description, corrections=[clibcorr])
        self.output()["json"].dump(cset.dict(exclude_unset=True), indent=4, formatter="json")


class BTagEfficiencyPlot(
    BTagEfficiencyBase,
    DatasetsMixin,
    CustomDefaultVariablesMixin,
    VariablesMixin,
    SelectorClassMixin,
    CalibratorClassesMixin,
    law.LocalWorkflow,
    PlotBase2D,
):
    resolution_task_cls = BTagEfficiency

    reqs = Requirements(BTagEfficiency=BTagEfficiency)

    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_2d.plot_2d",
        add_default_to_description=True,
    )

    dataset_group = luigi.Parameter(
        default="",
        description="the name of the label to print on the b-tagging efficiency plots to represent the dataset group.",
    )

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def create_branch_map(self):
        return [
            DotDict({"flav": flav, "wp": wp})
            for flav in self.flavours
            for wp in self.wps
        ]

    def requires(self):
        return self.reqs.BTagEfficiency.req(
            self,
            branch=-1,
            _exclude={"branches"},
        )

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["BTagEfficiency"] = self.requires_from_branch()

        return reqs

    def output(self):
        return [
            [
                self.target(name)
                for name in self.get_plot_names(
                    f"{self.tag_name}_eff__{self.flavours[self.branch_data.flav]}_{self.flav_name}"
                    f"__wp_{self.branch_data.wp}" +
                    (f"__err_{dr}" if dr != "central" else ""),
                )
            ]
            for dr in ["central", "down", "up"]
        ]

    def run(self):
        import hist
        import numpy as np

        # plot efficiency for each hadronFlavour and wp
        efficiency_hist = self.input()["collection"][0]["hist"].load(formatter="pickle")

        variable_insts = list(map(self.config_inst.get_variable, self.variables))
        variable_insts = sorted(variable_insts, key=efficiency_hist.axes.name.index)

        for i, sys in enumerate(["central", "down", "up"]):
            # create a dummy histogram dict for plotting with the first process
            # TODO change process name to the relevant process group
            h = efficiency_hist[{
                self.flav_name: hist.loc(self.branch_data.flav),
                f"{self.tag_name}_wp": self.branch_data.wp,
            }]

            h_sys = h[{"systematic": sys}]
            if sys != "central":
                h_sys -= h[{"systematic": "central"}].values()

            # create dummy process for plotting
            proc = od.Process(
                name=f"{self.datasets_repr}_{self.dataset_group}",
                id="+",
                label=self.dataset_group,
            )

            # create a dummy category for plotting
            cat = od.Category(
                name=self.flav_name,
                label=self.flavours[self.branch_data.flav],
            )

            # custom styling:
            label_values = np.round(h_sys.values() * 100, decimals=1)
            style_config = {"plot2d_cfg": {"cmap": "PiYG", "labels": label_values}}
            # call the plot function
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists={proc: h_sys},
                config_inst=self.config_inst,
                category_inst=cat.copy_shallow(),
                variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
                style_config=style_config,
                **self.get_plot_parameters(),
            )
            for p in self.output()[i]:
                p.dump(fig, formatter="mpl")
