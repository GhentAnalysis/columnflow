# coding: utf-8

"""
Inference model of the test template, restricted to what the CI test-data bundle provides: a single
tt_dl_powheg MC dataset, the inclusive category and fake data built from the TT process.
"""

from columnflow.inference import inference_model, ParameterType, ParameterTransformation


@inference_model
def test(self):

    #
    # categories
    #

    self.add_category(
        "incl",
        config_data={
            cfg.name: self.category_config_spec(
                category="incl",
                variable="jet1_pt",
            )
            for cfg in self.config_insts
        },
        # the bundle ships no data file, so fake data from TT
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
                mc_datasets=["tt_dl_powheg"],
            )
            for cfg in self.config_insts
        },
    )

    #
    # parameters
    #

    # lumi
    lumi = self.config_insts[0].x.luminosity
    for unc_name in lumi.uncertainties:
        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
            transformations=[ParameterTransformation.symmetrize],
        )

    # shape parameters, covering a config-level event weight shift (minbias_xs), a dataset-level
    # event weight shift (mur) and a selection dependent shift that re-runs the selection (jer)
    for shift_source in ["minbias_xs", "mur", "jer"]:
        self.add_parameter(
            shift_source,
            process="TT",
            type=ParameterType.shape,
            config_data={
                cfg.name: self.parameter_config_spec(shift_source=shift_source)
                for cfg in self.config_insts
            },
        )
