# coding: utf-8

"""
CI-only overlay that redirects a generated Ghent-template analysis to a self-contained test data
bundle instead of /cvmfs, DAS and the grid.

This module deliberately lives in the columnflow repository, outside of
``analysis_templates/``, so it can never be copied into a deployed analysis by
``create_analysis.sh``. The redirections it applies used to be baked directly into the template
config (reading ``CF_CI_TESTDATA`` at ``add_config`` time), which meant every user-deployed
analysis shipped with CI-specific branches. Here, the same effect is achieved at runtime by
monkeypatching the generated analysis's ``add_config`` function, driven entirely by the
``CF_CI_TESTDATA`` environment variable. See ``tests/ci/README.md`` for how the bundle referenced
by that variable is built and how this overlay is wired into the workflow.
"""

from __future__ import annotations

import os
import importlib
import functools

import law


logger = law.logger.get_logger(__name__)

# prefix used by the template config for all correctionlib inputs in a normal (non-CI) run
CVMFS_JSONPOG_PREFIX = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration"


def _redirect_cvmfs_paths(node: object, old_prefix: str, new_prefix: str, _ctx: str = "external_files") -> int:
    """
    Recursively walks a (possibly nested) mapping of external files, in-place rewriting any string
    (or first element of a tuple/list, i.e. the "(path, version)" pattern) that starts with
    *old_prefix* so that it starts with *new_prefix* instead. Returns the number of rewritten
    entries.
    """
    n_rewritten = 0

    if not isinstance(node, dict):
        return n_rewritten

    for key, value in node.items():
        ctx = f"{_ctx}.{key}"

        if isinstance(value, dict):
            n_rewritten += _redirect_cvmfs_paths(value, old_prefix, new_prefix, ctx)
        elif isinstance(value, (tuple, list)):
            new_value = list(value)
            changed = False
            for i, item in enumerate(new_value):
                if isinstance(item, str) and item.startswith(old_prefix):
                    new_value[i] = new_prefix + item[len(old_prefix):]
                    changed = True
            if changed:
                node[key] = tuple(new_value) if isinstance(value, tuple) else new_value
                n_rewritten += 1
                logger.debug(f"redirected {ctx} -> {node[key]}")
        elif isinstance(value, str) and value.startswith(old_prefix):
            node[key] = new_prefix + value[len(old_prefix):]
            n_rewritten += 1
            logger.debug(f"redirected {ctx} -> {node[key]}")

    return n_rewritten


def _patch_config(cfg: object, ci_data: str) -> None:
    """
    Post-processes a freshly built config object *cfg*, redirecting external files, dataset LFN
    resolution and btag dataset grouping to the CI test data bundle rooted at *ci_data*.
    """
    external_files = getattr(cfg.x, "external_files", None)
    if external_files is not None:
        n_rewritten = _redirect_cvmfs_paths(external_files, CVMFS_JSONPOG_PREFIX, f"{ci_data}/jsonpog")
        logger.info(f"redirected {n_rewritten} external_files entries under {CVMFS_JSONPOG_PREFIX} to {ci_data}/jsonpog")
    else:
        logger.warning(f"config '{cfg.name}' has no cfg.x.external_files; nothing to redirect")

    # serve a single local nano file instead of querying DAS via dasgoclient
    nano_file = os.path.join(ci_data, "nano", "tt_dl_powheg_2018_nano_v9.root")
    cfg.x.get_dataset_lfns = lambda task, key: [nano_file]
    # NO_STR (not None!): None is replaced by the cvmfs cmsset_default.sh sandbox in
    # columnflow/tasks/external.py, which is unreachable in CI
    cfg.x.get_dataset_lfns_sandbox = law.NO_STR
    logger.info(f"redirected cfg.x.get_dataset_lfns to a single local file: {nano_file}")

    # the lambda above serves exactly one file, but the branch map of every file-based workflow is
    # built from the dataset's declared n_files. Without clamping, branches >= 1 index past the end
    # of the lfn list and die with "IndexError: list index out of range" in iter_nano_files.
    for dataset in cfg.datasets:
        for info in dataset.info.values():
            info.n_files = 1
    logger.info(f"clamped n_files to 1 for {len(cfg.datasets)} datasets")

    # keep BTagEfficiency from fanning out over the full tt/dy dataset groups
    cfg.x.btag_dataset_groups = {"tt": ["tt_dl_powheg"]}
    logger.info(f"restricted cfg.x.btag_dataset_groups to {cfg.x.btag_dataset_groups}")


def apply(module_name: str, short_name: str) -> None:
    """
    Wraps ``<module_name>.config.config_<short_name>.add_config`` so that every config it builds is
    post-processed by :py:func:`_patch_config`. A no-op unless the ``CF_CI_TESTDATA`` environment
    variable is set, so importing and calling this function is always harmless in a normal,
    non-CI deployment.
    """
    ci_data = os.environ.get("CF_CI_TESTDATA")
    if not ci_data:
        return

    config_module_name = f"{module_name}.config.config_{short_name}"
    config_module = importlib.import_module(config_module_name)
    original_add_config = config_module.add_config

    @functools.wraps(original_add_config)
    def patched_add_config(*args, **kwargs):
        cfg = original_add_config(*args, **kwargs)
        _patch_config(cfg, ci_data)
        return cfg

    config_module.add_config = patched_add_config
    logger.info(f"patched {config_module_name}.add_config for CI test data at {ci_data}")
