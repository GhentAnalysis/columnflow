# coding: utf-8

"""
Helpers and utilities for working with columnar libraries (Ghent cms group)
"""

from __future__ import annotations
from typing import Literal
import law

__all__ = [
    "TetraVec", "safe_concatenate",
]

from columnflow.util import maybe_import
from columnflow.types import Sequence
from columnflow.columnar_util import remove_ak_column, has_ak_column

ak = maybe_import("awkward")
coffea = maybe_import("coffea")

logger = law.logger.get_logger(__name__)


def TetraVec(arr: ak.Array, keep: Sequence | str | Literal[-1] = -1) -> ak.Array:
    """
    create a Lorentz for fector from an awkward array with pt, eta, phi, and mass fields
    """
    mandatory_fields = ("pt", "eta", "phi", "mass")
    exclude_fields = ("x", "y", "z", "t")
    for field in mandatory_fields:
        assert hasattr(arr, field), f"Provided array is missing {field} field"
    if isinstance(keep, str):
        keep = [keep]
    elif keep == -1:
        keep = arr.fields
    keep = [*keep, *mandatory_fields]
    return ak.zip(
        {p: getattr(arr, p) for p in keep if p not in exclude_fields},
        with_name="PtEtaPhiMLorentzVector",
        behavior=coffea.nanoevents.methods.vector.behavior,
    )


def safe_concatenate(arrays, *args, **kwargs):
    n = len(arrays)
    if n > 2 ** 7:
        c1 = safe_concatenate(arrays[:n // 2], *args, **kwargs)
        c2 = safe_concatenate(arrays[n // 2:], *args, **kwargs)
        return ak.concatenate([c1, c2], *args, **kwargs)
    return ak.concatenate(arrays, *args, **kwargs)


def remove_obj_overlap(*arrays, objects=("Jet", "Electron", "Muon")):
    arrays = list(arrays)
    for obj in objects:
        for i1, c in enumerate(arrays[1:]):
            i1 += 1
            for i2, c0 in enumerate(arrays[:i1]):
                if has_ak_column(c0, obj) and has_ak_column(c, obj):
                    arrays[i2] = c0 = remove_ak_column(c0, obj)
    return arrays

def check_task_parquet_inputs(inputs, mode="check"):
    import pyarrow

    error = None
    if isinstance(inputs, (dict, tuple, list)):
        inputs = list(inputs.values()) if isinstance(inputs, dict) else inputs
        for k_inputs in inputs:
            error = check_task_parquet_inputs(k_inputs, mode=mode) or error
        return error
    elif (
        mode == "check"
        and isinstance(inputs, law.LocalFileTarget)
        and inputs.abspath.endswith("parquet")
        and inputs.exists()
    ):
        try:
            ak.metadata_from_parquet(inputs.abspath)
        except pyarrow.ArrowInvalid as e:
            # os.remove(inputs.abspath)
            return e
    elif mode == "remove":
        logger.error(f"removing {inputs}")
        inputs.remove()


def remove_corrupted_parquet(name, inputs):
    error = check_task_parquet_inputs(inputs)
    if error:
        logger.error(f"removing {name} inputs")
        check_task_parquet_inputs(inputs, mode="remove")
    return error
