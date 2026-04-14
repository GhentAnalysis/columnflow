import law
import order as od
import luigi
from collections import OrderedDict

from columnflow.tasks.plotting import PlotVariablesBaseSingleShift, PlotVariables1D
from columnflow.tasks.framework.plotting import PlotBase1D, PlotBase, PlotBase2D
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.util import DotDict

from columnflow.hist_util import add_missing_shifts


class PlotVariablesCatsPerProcessBase(PlotVariablesBaseSingleShift):

    exclude_index = True

    initial = luigi.Parameter(
        default="incl",
        description="Name of category that is considered as initial reference.",
    )

    def create_branch_map(self):
        cats = self.categories
        if self.initial not in cats:
            cats = [self.initial, *cats]

        out = [
            DotDict({
                "category": law.util.create_hash(cats),
                "categories": cats,
                "process": proc_name,
                "variable": var_name,
            })
            for proc_name in sorted(self.processes[0])
            for var_name in sorted(self.variables)
        ]
        return out

    def output(self):
        return {"plots": [
            self.local_target(name)
            for name in self.get_plot_names("plot")
        ]}

    @law.decorator.log
    @view_output_plots
    def run(self):
        import hist

        # get the shifts to extract and plot
        plot_shifts = law.util.make_list(self.get_plot_shifts())
        plot_shift_names = set(shift_inst.name for shift_inst in plot_shifts)

        # prepare config objects
        variable_tuple = self.variable_tuples[self.branch_data.variable]
        variable_insts = [
            self.config_inst.get_variable(var_name)
            for var_name in variable_tuple
        ]
        category_insts = [self.config_inst.get_category(c) for c in self.branch_data.categories]
        category_insts_leafs = [c.get_leaf_categories() or [c] for c in category_insts]

        # get assignment of processes to datasets and shifts
        config_process_map, process_shift_map = self.get_config_process_map()

        # filter for branch process
        config_process_map = {
            cfg: {p: v for p, v in process_map.items() if p.name == self.branch_data.process}
            for cfg, process_map in config_process_map.items()
        }
        process_shift_map = {
            p: v for p, v in process_shift_map.items()
            if p.name == self.branch_data.process
        }
        # sub_process_insts = [sub for sub, _, _ in process_inst.walk_processes(include_self=True)]


        # histogram data per category copy
        hists: dict[od.Config, dict[str, hist.Hist]] = {}
        with self.publish_step(f"plotting {self.branch_data.variable} for {process_inst.name}"):
            inputs = self.input() or self.workflow_input().merged_hists
            for i, (config, dataset_dict) in enumerate(inputs.items()):
                config_inst = self.config_insts[i]
                category_insts = [config_inst.get_category(c) for c in self.branch_data.categories]
                category_insts_leafs = [c.get_leaf_categories() or [c] for c in category_insts]

                # histogram data for process
                hists_config = {}
                for dataset, inp in dataset_dict:
                    dataset_inst = config_inst.get_dataset(dataset)
                    h_in = inp["collection"][0]["hists"].targets[self.branch_data.variable].load(formatter="pickle")

                    if h_in.empty():
                        continue

                    process_inst = config_inst.get_process(self.branch_data.process)
                    process_info = config_process_map[process_inst]


                    if dataset_inst not in process_info["dataset_proc_name_map"].keys():
                        continue

                    # select processes and reduce axis
                    h = h_in.copy()
                    h = h[{
                        "process": [
                            hist.loc(proc_name)
                            for proc_name in process_info["dataset_proc_name_map"][dataset_inst]
                            if proc_name in h.axes["process"]
                        ],
                    }]

                    add_missing_shifts(h, plot_shift_names, str_axis="shift", nominal_bin="nominal")

                    # add the histogram
                    if process_inst in hists_config:
                        hists_config[process_inst] += h
                    else:
                        hists_config[process_inst] = h

                hists[config_inst] = {
                    proc_inst: hists_config[proc_inst]
                    for proc_inst in sorted(
                        hists_config.keys(), key=list(config_process_map[config_inst].keys()).index,
                    )
                }

            # there should be hists to plot
            if not hists:
                raise Exception(
                    "no histograms found to plot; possible reasons:\n"
                    "  - requested variable requires columns that were missing during histogramming\n"
                    "  - selected --processes did not match any value on the process axis of the input histogram",
                )

            # update histograms using custom hooks
            hists = self.invoke_hist_hooks(
                hists,
                hook_kwargs={"category_name": self.branch_data.categories, "variable_name": self.branch_data.variable},
            )

            # merge configs
            if len(self.config_insts) != 1:
                process_memory = {}
                merged_hists = {}
                for _hists in hists.values():
                    for process_inst, h in _hists.items():
                        if process_inst.id in merged_hists:
                            merged_hists[process_inst.id] += h
                        else:
                            merged_hists[process_inst.id] = h
                            process_memory[process_inst.id] = process_inst

                process_insts = list(process_memory.values())
                hists = {process_memory[process_id]: h for process_id, h in merged_hists.items()}
            else:
                hists = hists[self.config_inst]
                process_insts = list(hists.keys())

            # axis selections
            for c, lcs in zip(category_insts, category_insts_leafs):
                hc = h[{
                    "category": [
                        hist.loc(c.id)
                        for c in lcs
                        if c.id in h.axes["category"]
                    ],
                }]

                # axis reductions
                hc = hc[{"category": sum}]

                # add the histsogram
                process_hists[c.name] = hc + process_hists[c.name]

            # there should be hists to plot
            if not all(process_hists.values()):
                raise Exception(
                    "no histograms found to plot; possible reasons:\n" +
                    "  - requested variable requires columns that were missing during histogramming\n" +
                    "  - selected --processes did not match any value on the process axis of the input histogram",
                )

            # update histograms using custom hooks
            hists = self.invoke_hist_hooks(process_hists)

            for cat in hists:
                if "process" in hists[cat].axes.name:
                    hists[cat] = hists[cat][{"process": sum}]

            # call the plot function
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists=hists,
                config_inst=self.config_inst,
                category_inst=process_inst.copy_shallow(),
                variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
                style_config={
                    "legend_cfg": {"title": process_inst.label},
                    "rax_cfg": {"ylabel": "Category / " + self.initial, "ylim": (.75, 1.25)},
                },
                initial=self.initial,
                **self.get_plot_parameters(),
            )

            # save the plot
            for outp in self.output()["plots"]:
                outp.dump(fig, formatter="mpl")


class PlotVariables1DCatsPerProcess(
    PlotVariablesCatsPerProcessBase,
    PlotBase1D,
):
    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_1d.plot_variable_variants",
    )


class PlotVariables2DMigration(
    PlotVariablesCatsPerProcessBase,
    PlotBase2D,
):
    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.cmsGhent.plot_functions_2d.plot_migration_matrices",
    )

    def create_branch_map(self):
        return [
            DotDict({
                "category": self.initial + "_" + cat_name,
                "categories": [self.initial, cat_name],
                "process": proc_name,
                "variable": var_name,
            })
            for cat_name in sorted(self.categories)
            for proc_name in sorted(self.processes)
            for var_name in sorted(self.variables)
        ]


class MultiVarMixin:

    def requires(self):
        variables = ",".join([",".join(self.variable_tuples[vr]) for vr in self.variables])
        reqs = {
            dataset: self.reqs.MergeHistograms.req(
                self,
                variables=variables,
                branch=-1,
                dataset=dataset,
                _exclude={"branches"},
                _prefer_cli={"variables"},
            )
            for dataset in self.datasets
        }
        return reqs

    def run_var(
            self,
            variable_inst: od.Variable,
            sub_process_insts: dict[od.Process, list[od.Process]],
            category_insts: list[od.Category],
            plot_shifts: list[od.Shift],
    ):
        import hist

        hists = {}
        process_insts = list(sub_process_insts)

        for dataset, inp in self.input().items():
            dataset_inst = self.config_inst.get_dataset(dataset)
            h_in = inp["collection"][0]["hists"].targets[variable_inst.name].load(formatter="pickle")

            # loop and extract one histogram per process
            for process_inst in process_insts:
                # skip when the dataset is already known to not contain any sub process
                if not any(
                        dataset_inst.has_process(sub_process_inst.name)
                        for sub_process_inst in sub_process_insts[process_inst]
                ):
                    continue

                # select processes and reduce axis
                h = h_in.copy()
                h = h[{
                    "process": [
                        hist.loc(p.id)
                        for p in sub_process_insts[process_inst]
                        if p.id in h.axes["process"]
                    ],
                }]
                h = h[{"process": sum}]

                # add the histogram
                if process_inst in hists:
                    hists[process_inst] += h
                else:
                    hists[process_inst] = h

        # there should be hists to plot
        if not hists:
            raise Exception(
                "no histograms found to plot; possible reasons:\n"
                "  - requested variable requires columns that were missing during histogramming\n"
                "  - selected --processes did not match any value on the process axis of the input histogram",
            )

        # update histograms using a custom hook
        hists = self.invoke_hist_hook(hists)

        # add new processes to the end of the list
        for process_inst in hists:
            if process_inst not in process_insts:
                process_insts.append(process_inst)

        # axis selections and reductions, including sorting by process order
        _hists = OrderedDict()
        for process_inst in sorted(hists, key=process_insts.index):
            h = hists[process_inst]
            # selections
            h = h[{
                "category": [
                    hist.loc(c.id)
                    for c in category_insts
                    if c.id in h.axes["category"]
                ],
                "shift": [
                    hist.loc(s.id)
                    for s in plot_shifts
                    if s.id in h.axes["shift"]
                ],
            }]
            # reductions
            h = h[{"category": sum}]
            # store
            _hists[process_inst] = h
        return _hists


class PlotVariables1DMultiVar(MultiVarMixin, PlotVariables1D):

    exclude_index = False

    plot_function = PlotVariables1D.plot_function.copy(
        default="columnflow.plotting.cmsGhent.plot_functions_1d.plot_multi_variables",
    )

    skip_ratio = PlotVariables1D.skip_ratio.copy(
        default=True,
    )

    @law.decorator.log
    @view_output_plots
    def run(self):
        # get the shifts to extract and plot
        plot_shifts = law.util.make_list(self.get_plot_shifts())

        # prepare config objects
        variable_tuple = self.variable_tuples[self.branch_data.variable]
        variable_insts = [
            self.config_inst.get_variable(var_name)
            for var_name in variable_tuple
        ]
        category_inst = self.config_inst.get_category(self.branch_data.category)
        leaf_category_insts = [category_inst]
        process_insts = list(map(self.config_inst.get_process, self.processes))
        sub_process_insts = {
            proc: [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        }

        # histogram data per variable
        hists = {}

        with self.publish_step(f"plotting {self.branch_data.variable} in {category_inst.name}"):

            # histogram data per variable

            for variable_inst in variable_insts:
                var_hists = self.run_var(variable_inst, sub_process_insts, leaf_category_insts, plot_shifts)
                var_hists = list(var_hists.values())
                hists[variable_inst] = sum(var_hists[1:], var_hists[0])

            # sort hists by varianle order
            hists = OrderedDict(
                (variable_inst, hists[variable_inst])
                for variable_inst in sorted(hists, key=variable_insts.index)
            )
            # call the plot function
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists=hists,
                config_inst=self.config_inst,
                category_inst=category_inst.copy_shallow(),
                **self.get_plot_parameters(),
            )

            # save the plot
            for outp in self.output()["plots"]:
                outp.dump(fig, formatter="mpl")


class PlotROC(MultiVarMixin, PlotVariablesBaseSingleShift):

    exclude_index = False

    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.cmsGhent.plot_functions_1d.plot_roc",
        add_default_to_description=True,
    )

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name})
            for cat_name in sorted(self.categories)
        ]

    def plot_parts(self) -> law.util.InsertableDict:
        parts = law.util.InsertableDict()
        parts["processes"] = f"proc_{self.processes_repr}"
        parts["category"] = f"cat_{self.branch_data.category}"
        parts["variable"] = f"var_{self.variables_repr}"

        if self.hist_hook not in ("", law.NO_STR, None):
            parts["hook"] = f"hook_{self.hist_hook}"

        return parts

    @law.decorator.log
    @view_output_plots
    def run(self):
        # get the shifts to extract and plot
        plot_shifts = law.util.make_list(self.get_plot_shifts())

        # prepare config objects
        variable_tuples = [self.variable_tuples[vr] for vr in self.variables]
        variable_insts_groups = [
            [self.config_inst.get_variable(var_name) for var_name in variable_tuple]
            for variable_tuple in variable_tuples
        ]
        category_inst = self.config_inst.get_category(self.branch_data.category)
        leaf_category_insts = [category_inst]
        process_insts = list(map(self.config_inst.get_process, self.processes))
        sub_process_insts = {
            proc: [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        }

        # histogram data per variable
        hists = {}

        with self.publish_step(f"plotting {self.variables} in {category_inst.name}"):

            # histogram data per variable

            for variable_insts in variable_insts_groups:
                for variable_inst in variable_insts:
                    var_hists = self.run_var(variable_inst, sub_process_insts, leaf_category_insts, plot_shifts)
                    var_hists = list(var_hists.values())
                    hists[variable_inst] = sum(var_hists[1:], var_hists[0])

            # call the plot function
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists=hists,
                config_inst=self.config_inst,
                category_inst=category_inst.copy_shallow(),
                variable_insts_groups=variable_insts_groups,
                **self.get_plot_parameters(),
            )

            # save the plot
            for outp in self.output()["plots"]:
                outp.dump(fig, formatter="mpl")
