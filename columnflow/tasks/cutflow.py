# coding: utf-8

"""
Tasks to be implemented: MergeSelectionMasks, PlotCutflow
"""

from collections import OrderedDict
from abc import abstractmethod

import luigi
import law
import order as od

from columnflow.tasks.framework.base import (
    Requirements, AnalysisTask, DatasetTask, ShiftTask, wrapper_factory, RESOLVE_DEFAULT,
)
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, VariablesMixin, CategoriesMixin, ChunkedIOMixin, MergeHistogramMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase1D, PlotBase2D, ProcessPlotSettingMixin, VariablePlotSettingMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.tasks.framework.parameters import last_edge_inclusive_inst
from columnflow.tasks.external import GetDatasetLFNs
from columnflow.tasks.selection import SelectEvents
from columnflow.tasks.calibration import CalibrateEvents
from columnflow.production import Producer
from columnflow.util import DotDict, dev_sandbox, maybe_import

from columnflow.hist_util import create_hist_from_variables

np = maybe_import("numpy")


class CreateCutflowHistograms(
    VariablesMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    ChunkedIOMixin,
    DatasetTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    # overwrite selector steps to use default resolution
    selector_steps = law.CSVParameter(
        default=(RESOLVE_DEFAULT,),
        description="a subset of steps of the selector to apply; uses all steps when empty; "
                    f"Set to {SelectorStepsMixin.selector_steps_all[0]} to apply all alphabetically."
                    "default: value of config.x.default_selector_steps",
        brace_expand=True,
        parse_empty=True,
    )

    steps_variable = od.Variable(
        name="step",
        aux={"axis_type": "strcategory"},
    )

    last_edge_inclusive = last_edge_inclusive_inst

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    selector_steps_order_sensitive = True

    initial_step = "Initial"

    default_variables = ("event", "cf_*")

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        GetDatasetLFNs=GetDatasetLFNs,
        CalibrateEvents=CalibrateEvents,
        SelectEvents=SelectEvents,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # store the normalization weight producer for MC
        self.norm_weight_producer = None
        if self.dataset_inst.is_mc:
            self.norm_weight_producer = Producer.get_cls("stitched_normalization_weights")(
                inst_dict=self.get_producer_kwargs(self),
            )

    # strategy for handling missing source columns when adding aliases on event chunks
    missing_column_alias_strategy = "original"

    # strategy for handling selector steps not defined by selectors
    missing_selector_step_strategy = luigi.ChoiceParameter(
        significant=False,
        default=law.config.get_default("analysis", "missing_selector_step_strategy", "raise"),
        choices=("raise", "ignore", "dummy"),
        description="how to handle selector steps that are not defined by the selector; if "
        "'raise', an exception will be thrown; if 'ignore', the selector step will be ignored; if "
        "'dummy' the output histogram will contain an entry for the step identical to the previous "
        "one; the default can be configured via the law config entry "
        "*missing_selector_step_strategy* in the *analysis* section; if no default is specified "
        "there, 'raise' is assumed",
    )

    def create_branch_map(self):
        # dummy branch map
        return [None]

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["lfns"] = self.reqs.GetDatasetLFNs.req(self)
        if not self.pilot:
            reqs["calibrations"] = [
                self.reqs.CalibrateEvents.req(self, calibrator=calibrator_inst.cls_name)
                for calibrator_inst in self.calibrator_insts
                if calibrator_inst.produced_columns
            ]
            reqs["selection"] = self.reqs.SelectEvents.req(self)
        else:
            # pass-through pilot workflow requirements of upstream task
            t = self.reqs.SelectEvents.req(self)
            reqs = law.util.merge_dicts(reqs, t.workflow_requires(), inplace=True)

        if self.dataset_inst.is_mc:
            reqs["normalization"] = self.norm_weight_producer.run_requires()

        return reqs

    def requires(self):
        reqs = {
            "lfns": self.reqs.GetDatasetLFNs.req(self),
            "calibrations": [
                self.reqs.CalibrateEvents.req(self, calibrator=calibrator_inst.cls_name)
                for calibrator_inst in self.calibrator_insts
                if calibrator_inst.produced_columns
            ],
            "selection": self.reqs.SelectEvents.req(self),
        }

        if self.dataset_inst.is_mc:
            reqs["normalization"] = self.norm_weight_producer.run_requires()

        return reqs

    # TODO: CreateHistograms has a @MergeReducedEventsUser.maybe_dummy here
    def output(self):
        return {
            "hists": self.target(f"histograms__vars_{self.variables_repr}__{self.branch}.pickle"),
        }

    @law.decorator.notify
    @law.decorator.log
    @law.decorator.localize(input=True, output=False)
    @law.decorator.safe_output
    def run(self):
        import numpy as np
        import awkward as ak
        from columnflow.columnar_util import Route, add_ak_aliases, mandatory_coffea_columns
        from columnflow.columnar_util import update_ak_array, has_ak_column
        from columnflow.hist_util import fill_hist

        # prepare inputs and outputs
        inputs = self.input()
        lfn_task = self.requires()["lfns"]

        # create a temp dir for saving intermediate files
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # get shift dependent aliases
        aliases = self.local_shift_inst.x("column_aliases", {})

        # setup the normalization weights producer
        if self.dataset_inst.is_mc:
            self.norm_weight_producer.run_setup(
                self.requires()["normalization"],
                self.input()["normalization"],
            )

        # define columns that need to be read
        read_columns = {"category_ids", "process_id"} | set(aliases.values())
        if self.dataset_inst.is_mc:
            read_columns |= self.norm_weight_producer.used_columns
        read_columns = {Route(c) for c in read_columns}

        # define steps
        steps = self.selector_steps

        # empty float array to use when input files have no entries
        empty_f32 = ak.Array(np.array([], dtype=np.float32))

        # prepare expressions
        expressions = {}
        for var_key, var_names in self.variable_tuples.items():
            variable_insts = [self.config_inst.get_variable(var_name) for var_name in var_names]

            # get the expression per variable and when a string, parse it to extract index lookups
            for variable_inst in variable_insts:
                expr = variable_inst.expression
                if isinstance(expr, str):
                    route = Route(expr)
                    def expr(events):
                        if len(events) == 0 and not has_ak_column(events, route):
                            return empty_f32
                        return route.apply(events, null_value=variable_inst.null_value)
                    read_columns.add(route)
                else:
                    # for variable_inst with custom expressions, read columns declared via aux key
                    read_columns |= set(variable_inst.x("inputs", []))
                expressions[variable_inst.name] = expr

        # prepare columns to load
        load_columns = read_columns | set(mandatory_coffea_columns)
        load_sel_columns = {Route("steps.*")}

        # prepare histograms
        histograms = {}
        def prepare_hists(steps):
            for var_key, var_names in self.variable_tuples.items():
                variable_insts = [self.config_inst.get_variable(var_name) for var_name in var_names]

                # create histogram of not already existing
                if var_key not in histograms:
                    histograms[var_key] = create_hist_from_variables(
                        self.steps_variable,
                        *variable_insts,
                        int_cat_axes=("category", "process", "shift"),
                    )

        # let the lfn_task prepare the nano file (basically determine a good pfn)
        [(lfn_index, input_file)] = lfn_task.iter_nano_files(self)

        # open the input file with uproot
        with self.publish_step("load and open ..."):
            nano_file = input_file.load(formatter="uproot")

        input_paths = [nano_file]
        input_paths.append(inputs["selection"]["results"].path)
        input_paths.extend([inp["columns"].path for inp in inputs["calibrations"]])
        if self.selector_inst.produced_columns:
            input_paths.append(inputs["selection"]["columns"].path)

        for (events, sel, *diffs), pos in self.iter_chunked_io(
            input_paths,
            source_type=["coffea_root"] + (len(input_paths) - 1) * ["awkward_parquet"],
            read_columns=[load_columns, load_sel_columns] + (len(input_paths) - 2) * [load_columns],
        ):

            # add the calibrated diffs and potentially new columns
            events = update_ak_array(events, *diffs)
            # add normalization weight
            if self.dataset_inst.is_mc:
                events = self.norm_weight_producer(events)

            # overwrite steps if not defined yet
            if steps == self.selector_steps_all:
                steps = sel.steps.fields

            # prepare histograms and exprepssions once
            if not histograms:
                prepare_hists([self.initial_step] + list(steps))

            # add aliases
            events = add_ak_aliases(
                events,
                aliases,
                remove_src=True,
                missing_strategy=self.missing_column_alias_strategy,
            )

            # pad the category_ids when the event is not categorized at all
            category_ids = ak.fill_none(ak.pad_none(events.category_ids, 1, axis=-1), -1)

            for var_key, var_names in self.variable_tuples.items():
                # helper to build the point for filling, except for the step which does
                # not support broadcasting
                def get_point(mask=Ellipsis):
                    if mask is True:
                        mask = Ellipsis
                    n_events = len(events) if mask is Ellipsis else ak.sum(mask)
                    point = {
                        "process": events.process_id[mask],
                        "category": category_ids[mask],
                        "shift": np.ones(n_events, dtype=np.int32) * self.global_shift_inst.id,
                        "weight": (
                            events.normalization_weight[mask]
                            if self.dataset_inst.is_mc
                            else np.ones(n_events, dtype=np.float32)
                        ),
                    }
                    for var_name in var_names:
                        point[var_name] = expressions[var_name](events)[mask]
                    return point

                # fill the raw point
                fill_data = get_point()
                fill_hist(
                    histograms[var_key],
                    fill_data,
                    fill_kwargs={"step": self.initial_step},
                    last_edge_inclusive=self.last_edge_inclusive,
                )
                # fill all other steps
                mask = True
                for step in steps:
                    if step not in sel.steps.fields:
                        if self.missing_selector_step_strategy == "raise":
                            raise ValueError(
                                f"step '{step}' is not defined by selector {self.selector}",
                            )
                        if self.missing_selector_step_strategy == "ignore":
                            continue
                    else:
                        # incrementally update the mask
                        mask = mask & sel.steps[step]

                    # fill the point
                    fill_data = get_point(mask)
                    fill_hist(
                        histograms[var_key],
                        fill_data,
                        fill_kwargs={"step": step},
                        last_edge_inclusive=self.last_edge_inclusive,
                    )

        # dump the histograms
        self.output()["hists"].dump(histograms, formatter="pickle")


CreateCutflowHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=CreateCutflowHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
)


class MergeCutflowHistograms(
    MergeHistogramMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    DatasetTask,
    RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateHistograms=CreateCutflowHistograms,
    )

    selector_steps_order_sensitive = True


MergeCutflowHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=MergeCutflowHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
)


class PlotCutflowBase(
    SelectorStepsMixin,
    CategoriesMixin,
    CalibratorsMixin,
    PlotBase,
    ShiftTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    selector_steps = CreateCutflowHistograms.selector_steps

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    exclude_index = True

    selector_steps_order_sensitive = True

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeCutflowHistograms=MergeCutflowHistograms,
    )

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "plot", f"datasets_{self.datasets_repr}")
        return parts


class PlotCutflow(
    PlotCutflowBase,
    PlotBase1D,
    ProcessPlotSettingMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_1d.plot_cutflow",
        add_default_to_description=True,
    )
    variable = luigi.Parameter(
        default=CreateCutflowHistograms.default_variables[0],
        significant=False,
        description="name of the variable to use for obtaining event counts; "
        f"default: '{CreateCutflowHistograms.default_variables[0]}'",
    )

    relative = luigi.BoolParameter(
        default=False,
        significant=False,
        description="plot cutflow as fraction of total at each step",
    )

    skip_initial = luigi.BoolParameter(
        default=False,
        significant=False,
        description="do not plot the event selection before applying any steps",
    )

    # upstream requirements
    reqs = Requirements(
        PlotCutflowBase.reqs,
        RemoteWorkflow.reqs,
    )

    def create_branch_map(self):
        # one category per branch
        if not self.categories:
            raise Exception(
                f"{self.__class__.__name__} task cannot build branch map when no categories are "
                "set",
            )

        return list(self.categories)

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["hists"] = [
            self.reqs.MergeCutflowHistograms.req(
                self,
                dataset=d,
                variables=(self.variable,),
                _exclude={"branches"},
            )
            for d in self.datasets
        ]
        return reqs

    def requires(self):
        return {
            d: self.reqs.MergeCutflowHistograms.req(
                self,
                branch=0,
                dataset=d,
                variables=(self.variable,),
                missing_selector_step_strategy="ignore",
            )
            for d in self.datasets
        }

    def plot_parts(self) -> law.util.InsertableDict:
        parts = super().plot_parts()
        parts["category"] = f"cat_{self.branch_data}"
        return parts

    def output(self):
        return {
            "plots": [self.target(name) for name in self.get_plot_names("cutflow")],
        }

    @law.decorator.notify
    @law.decorator.log
    @view_output_plots
    def run(self):
        import hist

        # copy process instances once so that their auxiliary data fields can be used as a storage
        # for process-specific plot parameters later on in plot scripts without affecting the
        # original instances
        fake_root = od.Process(
            name=f"{hex(id(object()))[2:]}",
            id="+",
            processes=list(map(self.config_inst.get_process, self.processes)),
        ).copy()
        process_insts = list(fake_root.processes)
        fake_root.processes.clear()

        # prepare config objects
        category_inst = self.config_inst.get_category(self.branch_data)
        leaf_category_insts = [category_inst] + (category_inst.get_leaf_categories() or [])
        sub_process_insts = {
            proc: [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        }

        # histogram data per process
        hists = {}

        with self.publish_step(f"plotting cutflow in {category_inst.name}"):
            for dataset, inp in self.input().items():
                dataset_inst = self.config_inst.get_dataset(dataset)
                h_in = inp["hists"][self.variable].load(formatter="pickle")

                # select shift
                if (n_shifts := len(h_in.axes["shift"])) != 1:
                    raise Exception(f"shift axis is supposed to only contain 1 bin, found {n_shifts}")
                h_in = h_in[{"shift": hist.loc(self.global_shift_inst.id)}]

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

                    if self.skip_initial:
                        h = h[{"step": self.selector_steps}]

                    # add the histogram
                    if process_inst in hists:
                        hists[process_inst] += h
                    else:
                        hists[process_inst] = h

            # there should be hists to plot
            if not hists:
                raise Exception("no histograms found to plot")

            total = sum(hists.values()).values() if self.relative else np.ones((len(self.selector_steps) + 1, 1))

            # axis selections and reductions, including sorting by process order
            _hists = OrderedDict()
            for process_inst in sorted(hists, key=process_insts.index):
                h = hists[process_inst]
                # selections
                h = h[{
                    "category": [
                        hist.loc(c.id)
                        for c in leaf_category_insts
                        if c.id in h.axes["category"]
                    ],
                }]
                # reductions
                h = h[{"category": sum, self.variable: sum}]
                # store
                _hists[process_inst] = h / total
            hists = _hists

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


PlotCutflowWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=PlotCutflow,
    enable=["configs", "skip_configs", "shifts", "skip_shifts"],
)


class PlotCutflowVariablesBase(
    VariablePlotSettingMixin,
    ProcessPlotSettingMixin,
    PlotCutflowBase,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    only_final_step = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, only create plots for the final selector step; default: False",
    )

    initial_step = "Initial"
    default_variables = ("cf_*",)

    exclude_index = True

    # upstream requirements
    reqs = Requirements(
        PlotCutflowBase.reqs,
        RemoteWorkflow.reqs,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # chosen selectors steps, including the initial one
        self.chosen_steps = [self.initial_step] + list(self.selector_steps)
        if self.only_final_step:
            del self.chosen_steps[:-1]

    def create_branch_map(self):
        if not self.categories:
            raise Exception(
                f"{self.__class__.__name__} task cannot build branch map when no categories are "
                "set",
            )
        if not self.variables:
            raise Exception(
                f"{self.__class__.__name__} task cannot build branch map when no variables are"
                "set",
            )

        return [
            DotDict({"category": cat_name, "variable": var_name})
            for cat_name in sorted(self.categories)
            for var_name in sorted(self.variables)
        ]

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["hists"] = [
            self.reqs.MergeCutflowHistograms.req(self, dataset=d, _exclude={"branches"})
            for d in self.datasets
        ]
        return reqs

    def requires(self):
        return {
            d: self.reqs.MergeCutflowHistograms.req(self, dataset=d, branch=0)
            for d in self.datasets
        }

    @abstractmethod
    def output(self):
        return

    @abstractmethod
    def run_postprocess(self, hists, category_inst, variable_insts):
        return

    @law.decorator.notify
    @law.decorator.log
    @view_output_plots
    def run(self):
        import hist

        # copy process instances once so that their auxiliary data fields can be used as a storage
        # for process-specific plot parameters later on in plot scripts without affecting the
        # original instances
        fake_root = od.Process(
            name=f"{hex(id(object()))[2:]}",
            id="+",
            processes=list(map(self.config_inst.get_process, self.processes)),
        ).copy()
        process_insts = list(fake_root.processes)
        fake_root.processes.clear()

        # prepare other config objects
        variable_tuple = self.variable_tuples[self.branch_data.variable]
        variable_insts = [
            self.config_inst.get_variable(var_name)
            for var_name in variable_tuple
        ]
        category_inst = self.config_inst.get_category(self.branch_data.category)
        leaf_category_insts = [category_inst] + (category_inst.get_leaf_categories() or [])
        sub_process_insts = {
            process_inst: [sub for sub, _, _ in process_inst.walk_processes(include_self=True)]
            for process_inst in process_insts
        }

        # histogram data per process copy
        hists = {}

        with self.publish_step(f"plotting {self.branch_data.variable} in {category_inst.name}"):
            for dataset, inp in self.input().items():
                dataset_inst = self.config_inst.get_dataset(dataset)
                h_in = inp["hists"][self.branch_data.variable].load(formatter="pickle")

                # select shift
                if (n_shifts := len(h_in.axes["shift"])) != 1:
                    raise Exception(f"shift axis is supposed to only contain 1 bin, found {n_shifts}")
                h_in = h_in[{"shift": hist.loc(self.global_shift_inst.id)}]

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
                raise Exception("no histograms found to plot")

            # axis selections and reductions, including sorting by process order
            _hists = OrderedDict()
            for process_inst in sorted(hists, key=process_insts.index):
                h = hists[process_inst]
                # selections
                h = h[{
                    "category": [
                        hist.loc(c.id)
                        for c in leaf_category_insts
                        if c.id in h.axes["category"]
                    ],
                }]
                # reductions
                h = h[{"category": sum}]
                # store
                _hists[process_inst] = h
            hists = _hists

            # call a postprocess function that produces outputs based on the implementation of the daughter task
            self.run_postprocess(
                hists=hists,
                category_inst=category_inst,
                variable_insts=variable_insts,
            )


class PlotCutflowVariables1D(
    PlotCutflowVariablesBase,
    PlotBase1D,
):
    plot_function = PlotBase.plot_function.copy(
        default=law.NO_STR,
        description=PlotBase.plot_function.description + "; the default is resolved based on the "
        "--per-plot parameter",
        allow_empty=True,
    )
    plot_function_processes = "columnflow.plotting.plot_functions_1d.plot_variable_per_process"
    plot_function_steps = "columnflow.plotting.plot_functions_1d.plot_variable_variants"

    per_plot = luigi.ChoiceParameter(
        choices=("processes", "steps"),
        default="processes",
        description="what to show per plot; choices: 'processes' (one plot per selection step, all "
        "processes in one plot); 'steps' (one plot per process, all selection steps in one plot); "
        "default: processes",
    )

    def plot_parts(self) -> law.util.InsertableDict:
        parts = super().plot_parts()
        parts["category"] = f"cat_{self.branch_data.category}"
        parts["variable"] = f"var_{self.branch_data.variable}"
        return parts

    def output(self):
        if self.per_plot == "processes":
            return {
                "plots": law.SiblingFileCollection({
                    s: [
                        self.local_target(name) for name in self.get_plot_names(
                            f"plot__step{i}_{s}__proc_{self.processes_repr}",
                        )
                    ]
                    for i, s in enumerate(self.chosen_steps)
                }),
            }

        # per_plot == "steps"
        return {
            "plots": law.SiblingFileCollection({
                p: [self.local_target(name) for name in self.get_plot_names(f"plot__proc_{p}")]
                for p in self.processes
            }),
        }

    def run_postprocess(self, hists, category_inst, variable_insts):
        import hist

        # resolve plot function
        if self.plot_function == law.NO_STR:
            self.plot_function = (
                self.plot_function_processes
                if self.per_plot == "processes"
                else self.plot_function_steps
            )

        if len(variable_insts) != 1:
            raise Exception(f"task {self.task_family} is only viable for single variables")

        outputs = self.output()["plots"]
        if self.per_plot == "processes":
            for step in self.chosen_steps:
                step_hists = OrderedDict(
                    (process_inst.copy_shallow(), h[{"step": hist.loc(step)}])
                    for process_inst, h in hists.items()
                    # skip missing steps
                    if step in h.axes["step"]
                )

                # call the plot function
                fig, _ = self.call_plot_func(
                    self.plot_function,
                    hists=step_hists,
                    config_inst=self.config_inst,
                    category_inst=category_inst.copy_shallow(),
                    variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
                    style_config={"legend_cfg": {"title": f"Step '{step}'"}},
                    **self.get_plot_parameters(),
                )

                # save the plot
                for outp in outputs[step]:
                    outp.dump(fig, formatter="mpl")

        else:  # per_plot == "steps"
            for process_inst, h in hists.items():
                process_hists = OrderedDict(
                    (step, h[{"step": hist.loc(step)}])
                    for step in self.chosen_steps
                    # skip missing steps
                    if step in h.axes["step"]
                )

                # call the plot function
                fig, _ = self.call_plot_func(
                    self.plot_function,
                    hists=process_hists,
                    config_inst=self.config_inst,
                    category_inst=category_inst.copy_shallow(),
                    variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
                    style_config={"legend_cfg": {"title": process_inst.label}},
                    **self.get_plot_parameters(),
                )

                # save the plot
                for outp in outputs[process_inst.name]:
                    outp.dump(fig, formatter="mpl")


class PlotCutflowVariables2D(
    PlotCutflowVariablesBase,
    PlotBase2D,
):
    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_2d.plot_2d",
        add_default_to_description=True,
    )

    def plot_parts(self) -> law.util.InsertableDict:
        parts = super().plot_parts()
        parts["processes"] = self.processes_repr
        parts["category"] = f"cat_{self.branch_data.category}"
        parts["variable"] = f"var_{self.branch_data.variable}"
        return parts

    def output(self):
        return {
            "plots": law.SiblingFileCollection({
                s: [self.local_target(name) for name in self.get_plot_names(f"plot__step{i}_{s}")]
                for i, s in enumerate(self.chosen_steps)
            }),
        }

    def run_postprocess(self, hists, category_inst, variable_insts):
        import hist

        outputs = self.output()["plots"]

        for step in self.chosen_steps:
            step_hists = OrderedDict(
                (process_inst.copy_shallow(), h[{"step": hist.loc(step)}])
                for process_inst, h in hists.items()
            )

            # call the plot function
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists=step_hists,
                config_inst=self.config_inst,
                category_inst=category_inst.copy_shallow(),
                variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
                style_config={"legend_cfg": {"title": f"Step '{step}'"}},
                **self.get_plot_parameters(),
            )

            # save the plot
            for outp in outputs[step]:
                outp.dump(fig, formatter="mpl")


class PlotCutflowVariablesPerProcess2D(
    law.WrapperTask,
    PlotCutflowVariables2D,
):
    # force this one to be a local workflow
    workflow = "local"

    # upstream requirements
    reqs = Requirements(
        PlotCutflowVariables2D.reqs,
        PlotCutflowVariables2D=PlotCutflowVariables2D,
    )

    def requires(self):
        return {
            process: self.reqs.PlotCutflowVariables2D.req(self, processes=(process,))
            for process in self.processes
        }
