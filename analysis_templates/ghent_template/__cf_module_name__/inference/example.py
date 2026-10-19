# coding: utf-8

"""
Example inference model.
"""

from collections import defaultdict

import order as od

from columnflow.inference import InferenceModel, inference_model, ParameterType, ParameterTransformation


@inference_model
def example(self):

    #
    # categories
    #

    self.add_category(
        "cat_incl",
        config_data={
            cfg.name: self.category_config_spec(
                category="incl",
                variable="jet1_pt",
                # to use real data instead of fake data, add e.g. data_datasets=["data_mu_*"] here and
                # remove data_from_processes below
            )
            for cfg in self.config_insts
        },
        # fake data from TT
        data_from_processes=["TT"],
        mc_stats=True,
    )

    #
    # processes
    #

    self.add_process(
        "TT",
        is_signal=True,
        config_data={
            cfg.name: self.process_config_spec(
                process="tt",
                # dataset names or patterns, e.g. add "tt_sl_powheg" to include semi-leptonic ttbar
                mc_datasets=["tt_dl_powheg"],
            )
            for cfg in self.config_insts
        },
    )

    #
    # parameters
    #

    # groups
    self.add_parameter_group("experiment")
    self.add_parameter_group("theory")

    # lumi
    # NOTE: luminosity uncertainties are assumed to be identical across all configs; only the
    # first config instance is used here (adjust if per-config luminosities are required)
    lumi = self.config_insts[0].x.luminosity
    for unc_name in lumi.uncertainties:
        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
            transformations=[ParameterTransformation.symmetrize],
        )
        self.add_parameter_to_group(unc_name, "experiment")

    # shape uncertainties from event weight variations, see the shifts and their column aliases in
    # config/shifts.py; the hist producer must declare these shifts (e.g. shifts_hist or all_weights)
    for shift_source, group in [
        ("minbias_xs", "experiment"),
        ("mur", "theory"),
        ("muf", "theory"),
    ]:
        self.add_parameter(
            shift_source,
            process="TT",
            type=ParameterType.shape,
            config_data={
                cfg.name: self.parameter_config_spec(shift_source=shift_source)
                for cfg in self.config_insts
            },
        )
        self.add_parameter_to_group(shift_source, group)

    # a custom asymmetric uncertainty that is converted from rate to shape
    self.add_parameter(
        "QCDscale_ttbar",
        process="TT",
        type=ParameterType.shape,
        transformations=[ParameterTransformation.effect_from_rate],
        effect=(0.5, 1.1),
    )
    self.add_parameter_to_group("QCDscale_ttbar", "theory")

    # use the histograms of collection shifts for parameters whose shift source they contain
    add_collection_shift_parameters(self)


@inference_model
def example_no_shapes(self):
    # same initialization as "example" above
    example.init_func.__get__(self, self.__class__)()

    #
    # remove all shape parameters
    #

    for category_name, process_name, parameter in self.iter_parameters():
        if parameter.type.is_shape or any(trafo.from_shape for trafo in parameter.transformations):
            self.remove_parameter(parameter.name, process=process_name, category=category_name)


def add_collection_shift_parameters(model: InferenceModel) -> None:
    """
    Adapts all parameters of *model* to collection shifts defined in its configs, i.e., shifts tagged
    ``is_collection`` whose ``subshifts`` histograms are all filled at once by the ``shifts_hist``
    hist producer (columnflow/histogramming/shifts.py). Does nothing when no collection shifts exist.

    Parameters whose shift source is part of a collection are marked dynamic, so that they do not
    request histograms of their own shift. Instead, one parameter per collection is added that
    requests the collection histograms, which contain the shifts of the dynamic parameters. This
    parameter itself has no effect in the datacard, since no histograms are stored for collection
    shifts. Note that datacards then require the ``shifts_hist`` hist producer.
    """
    # collection shift source -> config name -> shift sources contained in that collection
    collections: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for config_inst in model.config_insts:
        for shift_inst in config_inst.shifts:
            if shift_inst.has_tag("is_collection") and shift_inst.direction == od.Shift.UP:
                collections[shift_inst.source][config_inst.name] = {
                    config_inst.get_shift(subshift).source
                    for subshift in shift_inst.x("subshifts", [])
                }

    for collection_source, config_sources in collections.items():
        # mark contained parameters as dynamic and remember the processes they affect
        process_names = set()
        for _, process_name, parameter in model.iter_parameters():
            if any(
                config_name in parameter.config_data and parameter.config_data[config_name].shift_source in sources
                for config_name, sources in config_sources.items()
            ):
                parameter.is_dynamic = True
                process_names.add(process_name)

        # add the parameter requesting the collection histograms for the affected processes
        if process_names:
            model.add_parameter(
                collection_source,
                process=sorted(process_names),
                type=ParameterType.shape,
                config_data={
                    config_name: model.parameter_config_spec(shift_source=collection_source)
                    for config_name in config_sources
                },
            )
