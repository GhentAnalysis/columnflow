from __future__ import annotations

import law
import order as od
from collections import OrderedDict
from itertools import product

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, VariablesMixin,
    DatasetsProcessesMixin, SelectorMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase2D,
)

from columnflow.tasks.selection import MergeSelectionStats

from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.util import dev_sandbox, dict_add_strict
from columnflow.types import Any


class FixedWPEfficiencyBase(
    VariablesMixin,
    DatasetsProcessesMixin,
    SelectorMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
    PlotBase2D,
):

    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_2d.plot_2d",
        add_default_to_description=True,
    )

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeSelectionStats=MergeSelectionStats,
    )

    exclude_index = True

    tag_name = "btag"
    flav_name = "hadronFlavour"
    flavours = {0: "light", 4: "charm", 5: "bottom"}
    wps = ["L", "M", "T"]

    @classmethod
    def resolve_param_values(
            cls,
            params: law.util.InsertableDict[str, Any],
    ) -> law.util.InsertableDict[str, Any]:
        f"""
        Resolve values *params* and check against possible default values

        Check the values in *params* against the default value 
        ``"default_{cls.tag_name}_variables"`` in the current config inst.
        For more information, see
        :py:meth:`~columnflow.tasks.framework.base.ConfigTask.resolve_config_default_and_groups`.
        """
        redo_default_variables = False
        if "variables" in params:
            # when empty, use the config default
            if not params["variables"]:
                redo_default_variables = True

        params = super().resolve_param_values(params)

        config_inst = params.get("config_inst")
        if not config_inst:
            return params

        if redo_default_variables:
            # when empty, use the config default
            if config_inst.x(f"default_{cls.tag_name}_variables", ()):
                params["variables"] = tuple(config_inst.x(f"default_{cls.tag_name}_variables"))
            elif cls.default_variables:
                params["variables"] = tuple(cls.default_variables)
            else:
                raise AssertionError(f"define default {cls.tag_name} variables "
                                     f"in {cls.__class__} or config {config_inst.name}")

        return params

    def workflow_requires(self):
        reqs = super().workflow_requires()
        for d in self.datasets:
            reqs[d] = self.reqs.MergeSelectionStats.req(
                self,
                tree_index=0,
                branch=-1,
                dataset=d,
                _exclude=MergeSelectionStats.exclude_params_forest_merge,
            )
        return reqs

    def requires(self):
        return {
            d: self.reqs.MergeSelectionStats.req(
                self,
                tree_index=0,
                branch=-1,
                dataset=d,
                _exclude=MergeSelectionStats.exclude_params_forest_merge,
            )
            for d in self.datasets
        }

    def create_branch_map(self):
        # create a dummy branch map so that this task could be submitted as a job
        return {0: None}

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def output(self):
        return {
            "stats": self.target(".".join(
                self.get_plot_names(f"{self.tag_name}_efficiency")[0].split(".")[:-1],
            ) + ".json"),
            "plots": [
                [self.target(name)
                 for name in self.get_plot_names(
                    f"{self.tag_name}_eff__{flav}_{self.flav_name}"
                    f"__wp_{wp}",
                )]
                for flav in self.flavours.values()
                for wp in self.wps
            ],
        }

    def get_plot_parameters(self):
        # convert parameters to usable values during plotting
        params = super().get_plot_parameters()
        dict_add_strict(params, "legend_title", "Processes")
        return params

    @law.decorator.log
    def run(self):
        import hist
        import numpy as np
        import correctionlib
        import correctionlib.convert
        from columnflow.plotting.cmsGhent.plot_util import cumulate

        variable_insts = list(map(self.config_inst.get_variable, self.variables))

        # histogram for the tagged and all jets (combine all datasets)
        histogram = 0
        for dataset, inp in self.input().items():
            dataset_inst = self.config_inst.get_dataset(dataset)
            dt_process_insts = {process_inst for process_inst, _, _ in dataset_inst.walk_processes()}
            xsec = sum(
                process_inst.get_xsec(self.config_inst.campaign.ecm).nominal
                for process_inst in dt_process_insts
            )
            h_in = inp["collection"][0]["hists"].load(formatter="pickle")[f"{self.tag_name}_efficiencies"]
            histogram = histogram + h_in * xsec / inp["collection"][0]["stats"].load()["sum_mc_weight"]

        if not histogram:
            raise Exception(
                "no histograms found to plot; possible reasons:\n" +
                "  - requested variable requires columns that were missing during histogramming\n" +
                "  - selected --processes did not match any value on the process axis of the input histogram",
            )

        # combine tagged and inclusive histograms to an efficiency histogram
        cum_histogram = cumulate(histogram, direction="above", axis=f"{self.tag_name}_wp")
        incl = cum_histogram[{f"{self.tag_name}_wp": slice(0, 1)}].values()

        axes = OrderedDict(zip(cum_histogram.axes.name, cum_histogram.axes))
        axes[f"{self.tag_name}_wp"] = hist.axis.StrCategory(self.wps, name=f"{self.tag_name}_wp", label="working point")

        efficiency_hist = hist.Hist(*axes.values(), name=histogram.name, storage=hist.storage.Weight())
        efficiency_hist.view()[:] = cum_histogram[{f"{self.tag_name}_wp": slice(1, None)}].view()
        efficiency_hist = efficiency_hist / incl

        # save as correctionlib file
        efficiency_hist.label = "out"
        description = f"{self.tag_name} efficiencies of jets for {efficiency_hist.name} algorithm"
        clibcorr = correctionlib.convert.from_histogram(efficiency_hist)
        clibcorr.description = description

        cset = correctionlib.schemav2.CorrectionSet(schema_version=2, description=description, corrections=[clibcorr])
        self.output()["stats"].dump(cset.dict(exclude_unset=True), indent=4, formatter="json")
        # plot efficiency for each hadronFlavour and wp
        for i, (flav, wp) in enumerate(product(self.flavours, self.wps)):

            # create a dummy histogram dict for plotting with the first process
            hist_dict = OrderedDict((
                (self.config_inst.get_process(self.processes[-1]),
                 efficiency_hist[{
                     self.flav_name: hist.loc(flav),
                     f"{self.tag_name}_wp": wp,
                 }]),),
            )

            # create a dummy category for plotting
            cat = od.Category(name=self.flav_name, label=self.flavours[flav])

            # custom styling:
            label_values = np.around(
                efficiency_hist[{self.flav_name: hist.loc(flav)}].values() * 100, decimals=1)
            style_config = {"plot2d_cfg": {"cmap": "PiYG", "labels": label_values}}
            # call the plot function
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists=hist_dict,
                config_inst=self.config_inst,
                category_inst=cat.copy_shallow(),
                variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
                style_config=style_config,
                **self.get_plot_parameters(),
            )
            for p in self.output()["plots"][i]:
                p.dump(fig, formatter="mpl")


class BTagEfficiency(FixedWPEfficiencyBase):
    exclude_index = False


class LeptonMVAEfficiency(FixedWPEfficiencyBase):
    exclude_index = False

    tag_name = "lepton_mva"
    flav_name = "pdgId"
    flavours = {11: "electron", 13: "muon"}
    wps = ["VeryLoose", "Loose", "Medium", "Tight"]