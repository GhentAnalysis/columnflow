from __future__ import annotations
from typing import Callable, Sequence, Literal
from collections.abc import Collection

from columnflow.util import maybe_import
from columnflow.production.cmsGhent.trigger.Koopman_test import koopman_confint
import columnflow.production.cmsGhent.trigger.util as util

import numpy as np

hist = maybe_import("hist")

Hist = hist.Hist


def calc_stat(
    histograms: dict[str, Hist],
    trigger: str,
    ref_trigger: str,
    store_hists: dict,
    error_func: Callable[[float, float, float, float], tuple[float, float]] = koopman_confint,
) -> Hist:
    """
    Calculate statistical uncertainty on efficiency scale factors using confidence intervals.

    This function computes bin-by-bin statistical uncertainties on the scale factor (SF)
    derived from efficiency measurements in data and MC. The efficiency is computed as:

        eff = N(trigger & ref_trigger) / N(ref_trigger)

    The statistical uncertainty is evaluated using a confidence interval function (by default,
    `koopman_confint`) which calculates a lower and upper bound. The result is returned as a
    histogram with an added variation axis representing the down and up deviations from the
    nominal scale factor.

    Parameters
    ----------
    histograms : dict[str, Hist]
        Dictionary containing "data" and "mc" histograms with trigger and reference trigger counts.
    trigger : str
        Name of the trigger (numerator in the efficiency calculation).
    ref_trigger : str
        Name of the reference trigger (denominator in the efficiency calculation).
    store_hists : dict
        Dictionary for storing intermediate histograms (not used in this function).
    error_func : Callable[[float, float, float, float], tuple[float, float]], optional
        Function to compute the lower and upper bounds of the scale factor uncertainty.
        Inputs are: data_trigger_and_ref, data_ref, mc_trigger_and_ref, mc_ref.
        Defaults to `koopman_confint`.

    Returns
    -------
    Hist
        A histogram containing the statistical uncertainty band (down and up variations) on the
        scale factor, binned in all axes except the trigger and reference trigger axes.

    Notes
    -----
    - Assumes the input histograms contain integer counts or weights compatible with efficiency estimation.
    - Output histogram includes a systematic variation axis labeled "stat".
    """

    # Prepare output histogram for statistical uncertainty with all axes except trigger/ref_trigger
    out_hist = util.syst_hist(
        [ax for ax in histograms["data"].axes if ax.name not in (trigger, ref_trigger)],
        syst_name="stat",
    )

    # Loop over corresponding bins in both data and MC histograms, excluding the trigger axis
    for idx, hist_bins in util.loop_hists(histograms["data"], histograms["mc"], exclude_axes=trigger):
        # Skip bins with data that doesn't pass the reference trigger
        if not idx.pop(ref_trigger):
            continue

        # Extract counts in the bin: [data_trigger_and_ref, data_ref, mc_trigger_and_ref, mc_ref]
        inputs = [h[bn].value for h in hist_bins for bn in [1, sum]]

        # Compute lower and upper uncertainty bounds using the provided error function
        out_hist_idx = out_hist[idx]
        out_hist_idx.values()[:] = error_func(*inputs)

        # Store the result back into the output histogram
        out_hist[idx] = out_hist_idx.view()

    return out_hist


def calc_corr(
    histograms: dict[str, Hist],
    trigger: str,
    ref_trigger: str,
    store_hists: dict,
    corr_variables: Sequence[str] = tuple(),
    corr_func: Callable[[Hist], tuple[float, float]] = util.correlation_efficiency_bias,
    tag=None,
) -> Hist:
    """
        Calculate correlation bias in scale factor estimation due to correlated triggers in MC.

        This function computes the correlation bias between a given trigger and a reference trigger
        using MC histograms. It estimates how correlations in multi-dimensional phase space can bias
        scale factor (SF) calculations and returns a histogram representing the corresponding systematic
        uncertainty band (down and up variations).

        Parameters
        ----------
        histograms : dict[str, Hist]
            Dictionary containing histograms for different dataset types (e.g., "mc", "data").
        trigger : str
            The trigger of interest used for the numerator in efficiency calculations.
        ref_trigger : str
            The reference trigger used as the denominator in efficiency calculations.
        store_hists : dict
            Dictionary used to store intermediate histograms, such as the correlation bias histogram.
        corr_variables : Sequence[str], optional
            Names of the axes (variables) to preserve when reducing the histogram. The correlation
            bias will be binned in these variables. Defaults to an empty tuple (i.e., fully inclusive).
        corr_func : Callable[[Hist], tuple[float, float]], optional
            Function that calculates the correlation metric (central value and variance) from a projected
            histogram containing only the trigger and reference trigger axes. Defaults to
            `util.correlation_efficiency_bias`.
        tag : str, optional
            Optional label used to store the histogram in `store_hists`. If not provided, one is
            automatically generated based on `corr_variables`.

        Returns
        -------
        Hist
            A histogram representing the systematic uncertainty band (up and down variations) due to
            correlation bias, with the same axes as the input MC histogram and additional variation axis.

        Notes
        -----
        - The function works only on the "mc" entry in the input `histograms` dictionary.
        - The result can be used to propagate systematic uncertainties related to correlation bias in
          scale factor (SF) calculations.
        - The output is constructed using `util.syst_hist` and matches the axes of the original histogram.
        """

    # Get the MC histogram from input (we only use MC for correlation bias estimation)
    mc_hist = histograms["mc"]
    triggers = (trigger, ref_trigger)

    # Step 1: Create a histogram to store correlation values for every bin (before reducing dimensions)
    # This helps broadcast the result back across the full binning for uncertainty calculation
    unred_corr_hist = mc_hist[{t: sum for t in triggers}]

    # Step 2: Reduce the histogram keeping only the trigger axes and variables in which the correlation is binned
    mc_hist = util.reduce_hist(mc_hist, exclude=[*triggers, *corr_variables])

    # Step 3: Identify axes to keep for correlation histogram (i.e., variables in which the variables are binned)
    corr_vars = [ax.name for ax in mc_hist.axes if ax.name not in triggers]

    # Step 4: Prepare a histogram to store the computed correlation values (binned in corr_vars)
    if corr_vars:
        corr_hist = mc_hist.project(*corr_vars)
    else:
        corr_hist = hist.Hist.new.IntCategory([0]).Weight()
    corr_hist.name = "correlation bias" + (f"({', '.join(corr_vars)})" if corr_vars else "")
    corr_hist.label = (
        f"correlation bias for {trigger} trigger with reference {ref_trigger} "
        f"(binned in {', '.join(corr_vars)})" if corr_vars else "(inclusive)"
    )

    # Step 5: Loop over all bins in the reduced histogram (excluding trigger axes)
    for idx, hist_bin in util.loop_hists(mc_hist, exclude_axes=triggers):
        # Project to trigger axes only and compute trigger correlation using the user-defined function
        c = corr_func(hist_bin.project(*triggers))

        # Convert correlation output to a structured array with value and variance
        if isinstance(c, tuple):
            dtype = np.dtype([("value", float), ("variance", float)])
            c = np.array(corr_func(hist_bin.project(*triggers)), dtype=dtype)

        # Store the result in the binned correlation histogram
        corr_hist[idx] = c if idx else [c]

        # Also broadcast the same correlation value into the un-reduced histogram
        unred_corr_hist[idx] = np.full_like(unred_corr_hist[idx], c)

    # Step 6: Store the correlation histogram for optional plotting or later access
    if tag is None:
        tag = "corr" + ("" if not corr_vars else ("_" + "_".join(corr_vars)))
    store_hists[tag] = corr_hist

    # Step 7: Calculate nominal efficiency scale factors for both data and MC
    eff = {dt: util.calculate_efficiency(histograms[dt], *triggers) for dt in histograms}
    sf = eff["data"].values() / eff["mc"].values()

    # Step 8: Compute up/down variations by scaling SFs with the correlation bias factor
    sf_vars = [sf - sf * unred_corr_hist.values(), sf + sf * unred_corr_hist.values()]

    # Convert variations into a systematic histogram with variation axis
    sf_vars = util.syst_hist(unred_corr_hist.axes, syst_name=tag, arrays=sf_vars)

    return sf_vars


dev_funcs = {
    "max_dev_sym": lambda dev, idx: np.max(np.abs(dev), axis=idx),
    "max_dev": lambda dev, idx: (np.abs(np.min(dev, axis=idx)), np.abs(np.max(dev, axis=idx))),
    "std": lambda dev, idx: np.std(np.abs(dev), axis=idx),
}


def calc_auxiliary_unc(
    histograms: dict[str, Hist],
    trigger: str,
    ref_trigger: str,
    store_hists: dict,
    auxiliaries: list[str],
    apply_aux_to: Literal["data", "mc", "both"] = "both",
    dev_func: str | Callable[
        [np.ndarray, tuple[int]],  # an array, indices of auxilaray indices
        np.ndarray | tuple[np.ndarray, np.ndarray],  # symmetric or down, up
    ] = "max_dev_sym",
):
    """
        Calculate systematic uncertainty from auxiliary variables in scale factor (SF) computation.

        This function computes the deviation in data/MC efficiency scale factors due to auxiliary
        variables by comparing nominal and auxiliary efficiencies. It supports symmetric or asymmetric
        uncertainty evaluation using a specified deviation function.

        Parameters
        ----------
        histograms : dict[str, Hist]
            Dictionary mapping dataset types ("data", "mc") to their corresponding histograms.
        trigger : str
            The trigger of interest used for efficiency calculation.
        ref_trigger : str
            The reference trigger used as the denominator in efficiency calculation.
        store_hists : dict
            Dictionary for optionally storing intermediate histograms (not used internally here).
        auxiliaries : list[str]
            List of auxiliary axis names along which systematic variations are defined.
        apply_aux_to : {"data", "mc", "both"}, optional
            Specifies whether to apply auxiliary variations to data, MC, or both. Default is "both".
        dev_func : str or Callable, optional
            Function (or name of a registered function) to calculate deviations from the nominal SF.
            It takes the SF difference array and the indices of auxiliary axes and returns either a
            single array (symmetric uncertainty) or a tuple of arrays (downward and upward deviations).

        Returns
        -------
        Hist
            A histogram containing the systematic variation band (down and up) around the nominal SF,
            with the same axes as the nominal SF histogram.

        Notes
        -----
        - The returned histogram can be used as an input to uncertainty propagation or plotting.
        - The output is constructed using `util.syst_hist` and matches the axes of the original histogram.
    """

    triggers = (trigger, ref_trigger)
    if isinstance(dev_func, str):
        dev_func = dev_funcs[dev_func]

    nom_hist = {dt: util.reduce_hist(histograms[dt], reduce=auxiliaries) for dt in histograms}
    eff = {dt: util.calculate_efficiency(nom_hist[dt], *triggers) for dt in nom_hist}
    sf = eff["data"] / eff["mc"].values()

    # overwrite nominal with auxiliary
    apply_aux_to = ["data", "mc"] if apply_aux_to == "both" else [apply_aux_to]
    eff_aux = eff | {dt: util.calculate_efficiency(histograms[dt], *triggers) for dt in apply_aux_to}
    sf_aux = eff_aux["data"] / eff_aux["mc"].values()

    aux_idx = [sf_aux.axes.name.index(vr) for vr in auxiliaries]
    dev = dev_func(sf_aux.values() - np.expand_dims(sf.values(), axis=aux_idx), aux_idx)
    if not isinstance(dev, tuple):
        dev = (dev, dev)

    return util.syst_hist(
        sf.axes,
        syst_name="aux" + "_".join(auxiliaries),
        arrays=[sf.values() - dev[0], sf.values() + dev[1]]
    )
