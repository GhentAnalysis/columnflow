# coding: utf-8

"""
Collection of patches of underlying columnflow tasks.
"""

import os

import law
from columnflow.util import memoize


logger = law.logger.get_logger(__name__)


@memoize
def patch_bundle_repo_exclude_files():
    from columnflow.tasks.framework.remote import BundleRepo

    # add analysis-specific files to exclude, as absolute paths inside the analysis repo
    # ("docs", "tests", "data", "tmp", ".data", ".github" and ".setups" are already excluded by
    # default for both the cf and the analysis repo, so only genuinely new paths are added here)
    repo_path = lambda *p: os.path.join(os.environ["__cf_short_name_uc___BASE"], *p)
    BundleRepo.exclude_files += [repo_path("assets"), repo_path(".law")]

    logger.debug("patched exclude_files of cf.BundleRepo")


@memoize
def patch_all():
    patch_bundle_repo_exclude_files()
