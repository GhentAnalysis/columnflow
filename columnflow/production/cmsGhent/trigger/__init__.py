# coding: utf-8
"""
Producer that produces a trigger scalefactors
"""

from __future__ import annotations

import law
import order as od
from dataclasses import dataclass, field, replace
from collections.abc import Collection, Sequence
from columnflow.util import maybe_import, DotDict
from columnflow.production.cmsGhent.trigger.util import reduce_hist, collect_hist
from columnflow.types import Any, Iterable, Callable

from columnflow.production.cmsGhent.trigger.hist_producer import bundle_trigger_histograms, trigger_efficiency_hists # noqa
from columnflow.production.cmsGhent.trigger.sf_producer import bundle_trigger_weights, trigger_scale_factors # noqa


np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


@dataclass
class TriggerSFConfig:
    triggers: str | Iterable[str]
    ref_triggers: str | Iterable[str]
    variables: Sequence[str]
    datasets: Iterable[str]
    corrector_kwargs: dict[str, Any] = field(default_factory=dict)

    tag: str = "trig"
    ref_tag: str = "ref"
    sf_name: str = f"trig_sf"
    aux: dict = field(default_factory=dict)
    objects: list[str] = None  # list of objects used in the calculation: derived from the variables if None
    config_name: str = None
    main_variables: Sequence[str] = None

    get_sf_file: Callable = None
    get_no_trigger_selection: Callable = lambda results: results.x("event_no_trigger", None)
    event_mask_func: Callable = None
    event_mask_uses: set = field(default_factory=set)

    uncertainties: list = field(default_factory=list)
    _stat_func = None

    def __post_init__(self):

        # reformat self.trigger to tuple
        if isinstance(self.triggers, str):
            self.triggers = {self.triggers}
        elif not isinstance(self.triggers, set):
            self.triggers = set(self.triggers)

        # reformat self.ref_trigger to tuple
        if isinstance(self.ref_triggers, str):
            self.ref_triggers = {self.ref_triggers}
        elif not isinstance(self.ref_triggers, set):
            self.ref_triggers = set(self.ref_triggers)

        if not isinstance(self.datasets, set):
            self.datasets = set(self.datasets)

        self.x = DotDict(self.aux)
        if self.config_name is None:
            self.config_name = f"hlt_{self.tag.lower()}_ref_{self.ref_tag.lower()}"
        if self.main_variables is None:
            self.main_variables = self.variables
        self.main_variables = sorted(self.main_variables, key=self.variables.index)

        uncertainties, self.uncertainties = self.uncertainties, []
        for unc in uncertainties:
            self.uncertainty(unc)

    def copy(self, **changes):
        return replace(self, **changes)

    def event_mask(self, func: Callable[[ak.Array], ak.Array] = None, uses: set = None) -> None:
        """
        Decorator to wrap a function *func* that should be registered as :py:meth:`mask_func`
        which is used to calculate the mask that should be applied to the lepton

        The function should accept one positional argument:

            - *events*, an awkward array from which the inouts are calculate


        The decorator does not return the wrapped function.
        """

        def decorator(func: Callable[[ak.Array], dict[ak.Array]]):
            self.event_mask_func = func
            self.event_mask_uses = self.event_mask_uses | uses

        return decorator(func) if func else decorator

    def uncertainty(
        self,
        func: Callable =None,
        variables: Collection[str]=None,
        collect_mc_data=True,
        stat=False,
        **unc_kwargs,
    ):
        if variables is None:
            variables = self.main_variables

        def decorator(func: Callable):

            def decorated_func(
                histograms: dict[od.Dataset, hist.Hist],
                *args,
                **kwargs,
            ):
                vrs = [self.tag, self.ref_tag, *variables]
                if collect_mc_data:
                    histograms = collect_hist(histograms)
                histograms = {dt: reduce_hist(h, exclude=vrs) for dt, h in histograms.items()}
                kwargs |= unc_kwargs
                return func(histograms, self.tag, self.ref_tag, *args, **kwargs)
            if stat:
                self._stat_func = lambda hs, *args, **kwargs: func(hs, self.tag, self.ref_tag, *args, **kwargs)
            self.uncertainties.append(decorated_func)
            return func

        return decorator(func) if func else decorator

    @property
    def stat_unc(self):
        return self._stat_func





