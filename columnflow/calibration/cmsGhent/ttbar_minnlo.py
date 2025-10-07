"""
Code to add lepton MVA to NanoAOD
"""

import law

from columnflow.calibration import Calibrator, calibrator
from columnflow.util import maybe_import, four_vec
from columnflow.columnar_util import set_ak_column
from columnflow.columnar_util_Ghent import TetraVec
from columnflow.tasks.external import BundleExternalFiles

np = maybe_import("numpy")
ak = maybe_import("awkward")


def norm(X, mean, std, scale):
    if scale == "log":
        X = np.log(np.clip(X, a_min=1e-6, a_max=None))
    # recenter and renormalize
    return (X - mean) / np.where(std > 1e-2, std, 1)


@calibrator(
    uses=four_vec("GenPart", "pdgId"),
    produces={"ttbar_minnlo_weight"},
    sandbox="bash::$CF_BASE/sandboxes/venv_onnxruntime.sh",
    input_norm={
        # keys: 0 = ttbar, (-)6 = (anti)top
        # values: mean, std, scale
        "pt": {
            0: (3.6520673599656903, 1.0123402362573612, "log"),
            6: (4.595855742518925, 0.7101176940989488, "log"),
            -6: (4.5986175957604045, 0.7103218938891299, "log"),
        },
        "rapidity": {
            0: (0.0001718810581680775, 1.0362455506718102, "linear"),
            6: (0.00022746366634849002, 1.213207643109532, "linear"),
            -6: (0.00011712322394057398, 1.2076422016031159, "linear"),
        },
        "phi": {
            0: (2.8943571877384285e-05, 1.8139038706413384, "linear"),
            6: (-0.00028213870737636996, 1.8136544140703632, "linear"),
            -6: (0.0003628069129526392, 1.8139415747773364, "linear"),
        },
        "mass": {
            0: (6.21729978047307, 0.2771419580231537, "log"),
            6: (171.93706459943778, 6.9652037622153, "linear"),
            -6: (171.93691192651536, 6.9500586980501575, "linear"),
        },
    },
    sample_rb=0.855,
)
def ttbar_minnlo_reweighting_producer(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces the HVQ to MiNNLO reweighting values.
    Requires an external file in the config under ``ttbar_minnlo``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "ttbar_minnlo":  f"YOURDIRECTORY/mymodel12_13TeV_MiNNLO_afterShower.onnx",            },
        })

    Requires adding the environment venv_onnx which includes onnx to the analysis or config. E.g.

    analysis_inst.x.bash_sandboxes = [
        "$CF_BASE/sandboxes/cf.sh",
        "$CF_BASE/sandboxes/venv_onnxruntime.sh",
    ]

    """
    input = []
    sum_top = None
    for pdgId in [6, -6, 0]:
        # get the initial top quarks
        if not pdgId:
            top = sum_top
        else:
            top = events.GenPart[events.GenPart.pdgId == pdgId]
            top = top[top.hasFlags("isLastCopy")]
            top = TetraVec(top)
            # sum top quarks
            sum_top = top if sum_top is None else (sum_top + top)

        # inputs
        top_inp = [
            norm(top[inp], *norms[pdgId])
            for inp, norms in self.input_norm.items()
        ] + [np.full(len(top), pdgId / 10)]
        input.append(np.array(top_inp))

    # top, antitop, ttbar > ttbar, top, antitop
    input = np.roll(input, 1, axis=0)

    # (3, 6, N) > (N, 3, 6)
    input = np.rollaxis(input, -1)

    label_name = self.model.get_outputs()[0].name
    input_name = self.model.get_inputs()[0].name
    pred = self.model.run([label_name], {input_name: input.astype(np.float32)})[0]
    events = set_ak_column(events, f"ttbar_minnlo_weight", pred[:, 1] / pred[:, 0])

    return events


@ttbar_minnlo_reweighting_producer.requires
def ttbar_minnlo_reweighting_producer_requires(self: Calibrator, task: law.Task, reqs: dict) -> None:
    if "external_files" in reqs:
        return
    reqs["external_files"] = BundleExternalFiles.req(task)


@ttbar_minnlo_reweighting_producer.setup
def ttbar_minnlo_reweighting_producer_setup(
    self: Calibrator,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    bundle = reqs["external_files"]

    # create the xgboost predictor
    import onnxruntime

    file = bundle.files.ttbar_minnlo.path
    self.model = onnxruntime.InferenceSession(file)
