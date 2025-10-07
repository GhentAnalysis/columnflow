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
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@calibrator(
    uses=four_vec("GenPart", "pdgId"),
    produces={"hdamp_{up,down}"},
    sandbox="bash::$CF_BASE/sandboxes/venv_onnxruntime.sh",
    maxM=243.9517,
    default_hdamp=1.379,
)
def hdamp_reweighting_producer(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces the hdamp reweighting scores.
    Based on https://twiki.cern.ch/twiki/pub/CMS/MLReweighting/ImplementationCMSSW.pdf
    Requires an external file in the config under ``hdamp``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "hdamp": {
                "up": f"YOURDIRECTORY/mymodel12_hdamp_up_13TeV.onnx",
                "down": f"YOURDIRECTORY/mymodel12_hdamp_down_13TeV.onnx",
            },
        })

    The onnx files can be found on this twiki:
    https://twiki.cern.ch/twiki/bin/view/CMS/MLReweighting

    Requires adding the environment venv_onnx which includes onnx to the analysis or config. E.g.

    analysis_inst.x.bash_sandboxes = [
        "$CF_BASE/sandboxes/cf.sh",
        "$CF_BASE/sandboxes/venv_onnxruntime.sh",
    ]

    """
    input = []
    sum_top = None
    for pdgId in [6, -6]:
        # get the initial top quarks
        top = events.GenPart[events.GenPart.pdgId == pdgId][:, 0]
        top = TetraVec(top)

        # sum top quarks
        sum_top = top if sum_top is None else (sum_top + top)

        #
        top_inp = np.array([
            np.log10(top.pt),
            top.rapidity,
            top.phi,
            top.mass / self.maxM,
        ] + [
            np.full(len(top), cst)
            for cst in [
                {6: 0.1, -6: 0.2}[pdgId],
                self.default_hdamp,
            ]
        ])
        input.append(top_inp)

    # if pt of sum larger then 1000, no reweighting
    mask = sum_top.pt < 1000

    # (2, 6, N) > (N, 2, 6)
    input = np.rollaxis(np.array(input), -1)[mask]
    for variation, model in self.models.items():
        label_name = model.get_outputs()[0].name
        input_name = model.get_inputs()[0].name
        pred = model.run([label_name], {input_name: input.astype(np.float32)})[0]
        out = np.ones(len(events))
        out[mask] = pred[:, 0] / pred[:, 1]
        events = set_ak_column(events, f"hdamp_{variation}", out)

    return events


@hdamp_reweighting_producer.requires
def hdamp_reweighting_producer_requires(
    self: Calibrator,
    task: law.Task,
    reqs: dict,
    **kwargs,
) -> None:
    if "external_files" in reqs:
        return
    reqs["external_files"] = BundleExternalFiles.req(task)


@hdamp_reweighting_producer.setup
def hdamp_reweighting_producer_setup(
    self: Calibrator,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    bundle = reqs["external_files"]

    # create the xgboost predictor
    import onnxruntime

    self.models = {}
    for variation in ["up", "down"]:
        file = bundle.files.hdamp[variation].path
        self.models[variation] = onnxruntime.InferenceSession(file)
