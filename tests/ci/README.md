# CI test-data bundle

`.github/workflows/template_e2e.yaml` runs the `ghent_template` analysis end to end
(`cf.CalibrateEvents` → `cf.PlotVariables1D`) on a GitHub-hosted `ubuntu-latest` runner. That
runner has no CERN credentials, no VOMS/arc proxy, and — critically — no `/cvmfs` mount. Five of
the six leaf `cfg.x.external_files` entries in
`analysis_templates/ghent_template/__cf_module_name__/config/config___cf_short_name_lc__.py` are
plain `/cvmfs/...` paths, and there is no anonymous public mirror of the POG correctionlib JSONs
(the `gitlab.cern.ch` raw URLs redirect to an SSO login page, not the file). The NanoAOD input
itself normally comes from `dasgoclient` + a grid proxy, neither of which exists in CI either.

So the workflow downloads a small tarball from a public GitHub Release, exports its location as the
`CF_CI_TESTDATA` environment variable, and redirects the five cvmfs-backed entries at it via a
CI-only overlay (see "Redirecting the analysis at the bundle" below). The sixth entry,
`cfg.x.external_files["lumi"]["golden"]`, is deliberately left alone: it is an
`https://cms-service-dqmdc.web.cern.ch/...` URL, not a `/cvmfs/...` path, so
`_redirect_cvmfs_paths` never touches it, and `cf.BundleExternalFiles` fetches it live over the
network on every run (see "What is still fetched live" below) — the bundle is not fully
self-contained. **This document is how the tarball gets built.** It is a manual, one-time (or
once-per-bump) step that requires IIHE/cvmfs access — the CI job is useless without a bundle behind
the release asset it downloads, so read this fully before touching the workflow.

Do not try to "simplify" this back into fetching the JSONs from a `gitlab.cern.ch` raw URL at CI
run time — that was tried conceptually and rejected: unauthenticated requests to those raw URLs
return an HTML SSO login page, not JSON, which fails the corrector setup in a confusing way deep
inside `correctionlib`.

## Bundle layout

The workflow unpacks the tarball to `$RUNNER_TEMP/testdata` and exports that directory as
`CF_CI_TESTDATA`. `tests/ci/ci_config_patch.py` (see below) reads this env var and expects exactly
this layout:

```text
ci-testdata-v1.tar.gz
└── (unpacked root)
    ├── jsonpog/
    │   └── POG/
    │       ├── JME/2018_UL/jet_jerc.json.gz
    │       ├── EGM/2018_UL/electron.json.gz
    │       ├── MUO/2018_UL/muon_Z.json.gz
    │       ├── BTV/2018_UL/btagging.json.gz
    │       └── LUM/2018_UL/puWeights.json.gz
    └── nano/
        └── tt_dl_powheg_2018_nano_v9.root
```

Both the directory names (`jsonpog`, `nano`) and the NanoAOD filename
(`tt_dl_powheg_2018_nano_v9.root`) are read verbatim from `tests/ci/ci_config_patch.py` — if you
rename anything here, update that module (and vice versa), or `cf.CalibrateEvents` will fail to
find its inputs.

## Redirecting the analysis at the bundle

The template config (`config___cf_short_name_lc__.py`) is completely CI-free: it always reads
correctionlib inputs from `/cvmfs`, always resolves dataset LFNs via DAS, and never mentions
`CF_CI_TESTDATA`. That is deliberate — anything CI-specific baked into the template would ship into
every analysis a user deploys with `create_analysis.sh`, whether or not they ever run this
workflow.

Instead, the redirection lives entirely in `tests/ci/ci_config_patch.py`, a module in the
columnflow repository itself (outside `analysis_templates/`, so `create_analysis.sh` never copies
it into a generated analysis). The workflow (`.github/workflows/template_e2e.yaml`, step
"Apply CI-only config overlay") wires it in after instantiating the template, by:

1. copying `tests/ci/ci_config_patch.py` into the generated analysis package directory
   (`${CF_ANALYSIS_DIR}/${CF_ANALYSIS_DIR}/`), and
2. appending two lines to the generated package's `__init__.py` that import and call its `apply()`
   function.

`__init__.py` runs first when the analysis package is imported — before `create_analysis()` runs
and calls the config's `add_config()` — so `apply()` gets a chance to monkeypatch `add_config`
before it is ever invoked. If `CF_CI_TESTDATA` is unset, `apply()` is a no-op, so the exact same
module is harmless to import in a normal, non-CI deployment.

Once patched, every config built by `add_config()` gets post-processed to:

- recursively rewrite every `cfg.x.external_files` entry rooted at
  `/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration` to `$CF_CI_TESTDATA/jsonpog` instead
  (non-cvmfs entries, such as the golden JSON `https://` URL, are left untouched),
- set `cfg.x.get_dataset_lfns` to return the single trimmed NanoAOD file from the bundle instead of
  querying DAS,
- set `cfg.x.get_dataset_lfns_sandbox` to `law.NO_STR` (not `None` — see
  `columnflow/tasks/external.py`, where `None` falls back to sourcing the cvmfs
  `cmsset_default.sh`, which is unreachable in CI),
- clamp `n_files` to 1 for every dataset info: branch maps of file-based workflows are built from
  the dataset's *declared* `n_files`, so without this, branches ≥ 1 index past the end of the
  single-entry LFN list above and fail with `IndexError: list index out of range` in
  `iter_nano_files`, and
- restrict `cfg.x.btag_dataset_groups` to just the dataset used in CI, so `BTagEfficiency` does not
  fan out over a dataset group whose other members were never selected/reduced.

## Declaring the runner a local environment

Redirecting the *inputs* is not enough on its own: `setup.sh` detects `GITHUB_ACTIONS=true` and
sets `CF_LOCAL_ENV=false`, and `cf.GetDatasetLFNs` and `cf.BundleExternalFiles` both refuse to run
in a non-local environment. Since the overlay above has already replaced every DAS/grid/cvmfs
access with a local file, the runner really is the local environment steering this run, so each
law step of the workflow sources `tests/ci/setup_ci_env.sh` instead of `setup.sh` directly. That
script sources `setup.sh` and then re-exports `CF_LOCAL_ENV=true`; its header explains why the
override comes *after* sourcing and why sandboxed tasks are unaffected.

## What is still fetched live

The bundle is *not* fully self-contained. Exactly one `cfg.x.external_files` entry,
`cfg.x.external_files["lumi"]["golden"]` (the golden-JSON luminosity certification file), is an
`https://cms-service-dqmdc.web.cern.ch/...` URL rather than a `/cvmfs/...` path, so
`_redirect_cvmfs_paths` in `tests/ci/ci_config_patch.py` deliberately leaves it untouched (see
`_patch_config`). `cf.BundleExternalFiles` therefore downloads it from that CERN service over the
network on **every** run — `data/cf_store` is not cached between CI runs, so there is no
run-to-run reuse. The consequence: an outage or slowdown of that CERN service reddens this job on
PRs that have nothing to do with the change being tested. If that becomes a recurring problem, the
golden JSON is the obvious next file to fold into a future `ci-testdata-v2` bundle (see "Bumping
the bundle version" below) — it is small and rarely changes.

## Coverage gaps

Two things the overlay deliberately does not reproduce faithfully, so a green run should not be
read as full coverage:

- **`cfg.x.btag_dataset_groups` is cut to a single dataset** (`_patch_config` restricts it to
  `{"tt": ["tt_dl_powheg"]}`). The multi-dataset merge path inside `BTagEfficiency` — combining
  several datasets that belong to the same process group — is therefore never exercised by this
  job, even though it is real production code used outside CI.
- **`cmsdb` is added as an unpinned, moving branch.** `create_analysis.sh` adds it via
  `git submodule add -b "${fetch_cmsdb_branch}" ...` with `fetch_cmsdb_branch="GhentAnalysis/master"`
  (`create_analysis.sh:33`) — a branch, not a pinned SHA. A commit landing on that branch can
  therefore redden a columnflow PR that never touched `cmsdb`, and a run that was green once is not
  guaranteed to stay reproducible later. Pinning it to a SHA would fix this but is a maintainer
  call (it trades reproducibility for staying in sync with `cmsdb` development), so it is left
  unpinned here — just be aware of it when this job fails for no apparent reason.

## 1. Copy the correctionlib JSONs (`jsonpog/`)

`jsonpog/` is a straight copy of the matching subpaths from
`/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration` — the same tree the template reads from
in a normal (non-CI) run. Only the five files actually referenced by `cfg.x.external_files` for the
2018 UL campaign are needed:

- `POG/JME/2018_UL/jet_jerc.json.gz`
- `POG/EGM/2018_UL/electron.json.gz`
- `POG/MUO/2018_UL/muon_Z.json.gz`
- `POG/BTV/2018_UL/btagging.json.gz`
- `POG/LUM/2018_UL/puWeights.json.gz`

From a machine with the cvmfs mount available (e.g. an IIHE interactive node):

```bash
mkdir -p build/jsonpog/POG/{JME,EGM,MUO,BTV,LUM}/2018_UL
src=/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG

cp "${src}/JME/2018_UL/jet_jerc.json.gz"   build/jsonpog/POG/JME/2018_UL/
cp "${src}/EGM/2018_UL/electron.json.gz"   build/jsonpog/POG/EGM/2018_UL/
cp "${src}/MUO/2018_UL/muon_Z.json.gz"     build/jsonpog/POG/MUO/2018_UL/
cp "${src}/BTV/2018_UL/btagging.json.gz"   build/jsonpog/POG/BTV/2018_UL/
cp "${src}/LUM/2018_UL/puWeights.json.gz"  build/jsonpog/POG/LUM/2018_UL/
```

Do not add other years, `vfp` postfixes, or POG subsystems — the CI config only ever runs the 2018
UL campaign, and every extra file only bloats the release asset.

## 2. Trim the NanoAOD file (`nano/`)

Take a `tt_dl_powheg` 2018 UL NanoAODv9 file (found via DAS/dasgoclient on a machine with grid
access) and cut it down to a single file of roughly 66000 events. **Keep every branch** — do not
select a subset. The template reads `Jet`, `Electron`, `Muon`, `MET`, `Pileup`, `LHE*`,
`genWeight`, and various trigger bits (`HLT_*`) across calibration, selection, and production;
dropping any of these surfaces later as a confusing `KeyError`/`FieldNotFoundError` several tasks
downstream of where the branch was actually needed, not at read time. Keeping the full branch list
costs a bit of disk space but saves debugging time.

```python
import uproot

# any single NanoAODv9 file for tt_dl_powheg, 2018 UL, e.g. resolved via dasgoclient + xrootd
src = "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/.../NANOAODSIM/.../0000/xxxx.root"
n_events = 66000

with uproot.open(src) as fin:
    events = fin["Events"]
    # library="ak" preserves the jagged (per-object) structure of collections like Jet, Electron, ...
    arrays = events.arrays(library="ak", entry_stop=n_events)

with uproot.recreate("tt_dl_powheg_2018_nano_v9.root") as fout:
    fout["Events"] = arrays
```

Move the result into `nano/tt_dl_powheg_2018_nano_v9.root` inside the build directory from step 1.

## 3. Package and publish the release

```bash
cd build
tar czf ../ci-testdata-v1.tar.gz jsonpog nano
cd ..

gh release create ci-testdata-v1 ci-testdata-v1.tar.gz -R GhentAnalysis/columnflow \
  --title "CI test data v1" \
  --notes "Trimmed tt_dl_powheg 2018 UL NanoAOD (~66k events) + 2018 UL correctionlib JSONs, used by .github/workflows/template_e2e.yaml"
```

The release (and therefore the asset) **must be public**. The workflow downloads it with a plain
`curl`, with no `GITHUB_TOKEN` or other credential attached — if the release is private or draft,
the download 404s (or, worse, silently returns an HTML error page that `tar` then fails to unpack
in a way that's harder to diagnose than a clean 404).

## Bumping the bundle version

The tag (`ci-testdata-v1`), the workflow's cache key, and the download URL all encode the version
string and must be changed together:

- `.github/workflows/template_e2e.yaml`: the `actions/cache@v4` step for the test-data bundle
  (`key: ci-testdata-v1`) and the `curl` URL
  (`.../releases/download/ci-testdata-v1/ci-testdata-v1.tar.gz`).
- The release tag itself, created with `gh release create <new-tag> ...` above.

Bump the version (e.g. to `ci-testdata-v2`) whenever the bundle contents change — reusing an
existing tag/cache key for different contents means the `actions/cache@v4` step in CI will keep
serving the old, cached bundle to runners that already have it cached, silently skipping your
update.
