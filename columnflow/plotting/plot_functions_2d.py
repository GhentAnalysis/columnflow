# coding: utf-8

"""
Example 2d plot functions.
"""

from __future__ import annotations

from collections import OrderedDict

import law

from columnflow.util import maybe_import
from columnflow.plotting.plot_all import make_plot_2d
from columnflow.plotting.plot_util import (
    remove_residual_axis,
    apply_variable_settings,
    apply_process_settings,
    apply_density_to_hists,
    prepare_plot_config_2d,
    prepare_style_config_2d,
)

hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")


def plot_2d(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool | None = False,
    zscale: str | None = "",
    # z axis range
    zlim: tuple | None = None,
    # how to handle bins with values outside the z range
    extremes: str | None = "",
    # colors to use for marking out-of-bounds values
    extreme_colors: tuple[str] | None = None,
    colormap: str | None = "",
    skip_legend: bool = False,
    cms_label: str = "wip",
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    # remove shift axis from histograms
    remove_residual_axis(hists, "shift")

    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_process_settings(hists, process_settings)
    hists = apply_density_to_hists(hists, density)

    # how to handle yscale information from 2 variable insts?
    if not zscale:
        zscale = "log" if (variable_insts[0].log_y or variable_insts[1].log_y) else "linear"


    # how to handle bin values outside plot range
    if not extremes:
        extremes = "color"

    # add all processes into 1 histogram
    h_sum = sum(list(hists.values())[1:], list(hists.values())[0].copy())
    if shape_norm:
        h_sum = h_sum / h_sum.sum().value

    # if requested mask bins without any entries (variance == 0)
    h_view = h_sum.view()

    h_view.value[h_view.variance == 0] = np.nan

    # check histogram value range
    vmin, vmax = np.nanmin(h_sum.values()), np.nanmax(h_sum.values())
    vmin, vmax = np.nan_to_num([vmin, vmax], 0)

    # default to full z range
    if zlim is None:
        zlim = ("min", "max")

    # resolve string specifiers like "min", "max", etc.
    zlim = tuple(reduce_with(lim, h_sum.values()) for lim in zlim)

    # if requested, hide or clip bins outside specified plot range
    if extremes == "hide":
        h_view.value[h_view.value < zlim[0]] = np.nan
        h_view.value[h_view.value > zlim[1]] = np.nan
    elif extremes == "clip":
        h_view.value[h_view.value < zlim[0]] = zlim[0]
        h_view.value[h_view.value > zlim[1]] = zlim[1]

    # update histogram values from view
    h_sum[...] = h_view

    # choose appropriate colorbar normalization
    # based on scale type and histogram content

    # log scale (turning linear for low values)
    if zscale == "log":
        # use SymLogNorm to correctly handle both positive and negative values
        cbar_norm = mpl.colors.SymLogNorm(
            vmin=zlim[0],
            vmax=zlim[1],
            # TODO: better heuristics?
            linscale=1.0,
            linthresh=max(0.05 * min(abs(zlim[0]), abs(zlim[1])), 1e-3),
        )

    plot_config = prepare_plot_config_2d(
        hists,
        shape_norm=shape_norm,
        zscale=zscale,
        zlim=zlim,
        extremes=extremes,
        extreme_colors=extreme_colors,
        colormap=colormap,
    )

    default_style_config = prepare_style_config_2d(
        config_inst=config_inst,
        category_inst=category_inst,
        process_insts=list(hists.keys()),
        variable_insts=variable_insts,
        cms_label=cms_label,
    )


    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    if skip_legend:
        del style_config["legend_cfg"]

    return make_plot_2d(plot_config, style_config)
