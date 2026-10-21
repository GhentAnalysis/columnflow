import order as od

from columnflow.util import maybe_import, call_once_on_config

np = maybe_import("numpy")
ak = maybe_import("awkward")

from columnflow.columnar_util import EMPTY_FLOAT


@call_once_on_config()
def add_variables(config: od.Config) -> None:
    """
    Adds variables to a *config* that are produced as part of the `features` producer.
    """
    config.add_variable(
        name="event",
        expression=lambda events: np.zeros(len(events), dtype=int),
        binning=(1, 0, 1),
        x_title="Event number",
        x_discrete=True,
    )
    config.add_variable(
        name="n_jet",
        expression="n_jet",
        binning=(6, 1, 7),
        x_title="Number of jets",
        x_discrete=True,
    )
    config.add_variable(
        "jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, 0, 300),
        x_title=r"leading Jet $p_T$",
        unit="GeV",
    )
    config.add_variable(
        "genTop_pt",
        expression="genTop.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 300),
        x_title="Gen-level Top pT",
        unit="GeV",
    )
    config.add_variable(
        "jet1_eta",
        expression="Jet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(25, -2.5, 2.5),
        x_title=r"leading Jet $\eta$",
    )
    config.add_variable(
        name="ht",
        expression="ht",
        binning=(20, 0, 1000),
        x_title=r"$H_T$",
        unit="GeV",
    )
    config.add_variable(
        name="n_bjet",
        expression="n_bjet",
        binning=(4, 0, 4),
        x_title="Number of b-tagged jets",
        x_discrete=True,
    )

    # cutflow variable, filled from the cutflow.* columns written during SelectEvents
    config.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        null_value=EMPTY_FLOAT,
        binning=(30, 0, 300),
        x_title=r"leading Jet $p_T$",
        unit="GeV",
    )
