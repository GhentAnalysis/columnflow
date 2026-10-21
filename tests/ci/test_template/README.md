# __cf_analysis_name__ (columnflow test template)

This analysis was generated from the columnflow **test template** (`tests/ci/test_template` in the
GhentAnalysis/columnflow repository). It is not a starting point for a physics analysis; use the
`ghent_template` flavor of `create_analysis.sh` for that.

The test template is derived from `analysis_templates/ghent_template` and extends it for wider
coverage of columnflow tasks in CI (`.github/workflows/test_template_e2e.yaml`):

- all outputs are stored locally, chunks are small (`chunked_io_chunk_size: 20000`),
- the default hist producer is `all_weights`, so event weights and their shifts are applied,
- the `default` selector declares the `jer` shifts, making `jer` a selection dependent shift,
- the producer is split into `features` and `event_weights` (`default` combines both),
- the `test` inference model only uses what the CI test-data bundle provides (`tt_dl_powheg`,
  category `incl`, fake data from TT),
- the broken `example` ML model, hist producer and inference model of the ghent_template are
  removed.

The configuration still reads its inputs from `/cvmfs` and DAS, so the analysis also runs on a
machine with cvmfs and grid access. In CI, `tests/ci/ci_config_patch.py` redirects these to the
CI test-data bundle. See `tests/ci/README.md` for details.
