# CI test-data bundle

`.github/workflows/template_e2e.yaml` runs the `ghent_template` analysis end to end
(`cf.CalibrateEvents` → `cf.PlotVariables1D`) on a GitHub-hosted `ubuntu-latest` runner. That
runner has no CERN credentials, no VOMS/arc proxy, and — critically — no `/cvmfs` mount. Five of
the seven `cfg.x.external_files` entries in
`analysis_templates/ghent_template/__cf_module_name__/config/config___cf_short_name_lc__.py` are
plain `/cvmfs/...` paths, and there is no anonymous public mirror of the POG correctionlib JSONs
(the `gitlab.cern.ch` raw URLs redirect to an SSO login page, not the file). The NanoAOD input
itself normally comes from `dasgoclient` + a grid proxy, neither of which exists in CI either.

So the workflow does not talk to CVMFS, DAS, or the grid at all. Instead it downloads a small,
self-contained tarball from a public GitHub Release and points the analysis config at it via the
`CF_CI_TESTDATA` environment variable. **This document is how that tarball gets built.** It is a
manual, one-time (or once-per-bump) step that requires IIHE/cvmfs access — the CI job is useless
without a bundle behind the release asset it downloads, so read this fully before touching the
workflow.

Do not try to "simplify" this back into fetching the JSONs from a `gitlab.cern.ch` raw URL at CI
run time — that was tried conceptually and rejected: unauthenticated requests to those raw URLs
return an HTML SSO login page, not JSON, which fails the corrector setup in a confusing way deep
inside `correctionlib`.

## Bundle layout

The workflow unpacks the tarball to `$RUNNER_TEMP/testdata` and exports that directory as
`CF_CI_TESTDATA`. The template config reads this env var directly (see `config___cf_short_name_lc__.py`,
around the `external_files` block) and expects exactly this layout:

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
        └── tt_dl_powheg_2018_nano_v9_2k.root
```

Both the directory names (`jsonpog`, `nano`) and the NanoAOD filename
(`tt_dl_powheg_2018_nano_v9_2k.root`) are read verbatim from the config file — if you rename
anything here, update the config (and vice versa), or `cf.CalibrateEvents` will fail to find its
inputs.

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
access) and cut it down to roughly 2000 events. **Keep every branch** — do not select a subset. The
template reads `Jet`, `Electron`, `Muon`, `MET`, `Pileup`, `LHE*`, `genWeight`, and various trigger
bits (`HLT_*`) across calibration, selection, and production; dropping any of these surfaces later
as a confusing `KeyError`/`FieldNotFoundError` several tasks downstream of where the branch was
actually needed, not at read time. Keeping the full branch list costs a bit of disk space but saves
debugging time.

```python
import uproot

# any single NanoAODv9 file for tt_dl_powheg, 2018 UL, e.g. resolved via dasgoclient + xrootd
src = "root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/.../NANOAODSIM/.../0000/xxxx.root"
n_events = 2000

with uproot.open(src) as fin:
    events = fin["Events"]
    # library="ak" preserves the jagged (per-object) structure of collections like Jet, Electron, ...
    arrays = events.arrays(library="ak", entry_stop=n_events)

with uproot.recreate("tt_dl_powheg_2018_nano_v9_2k.root") as fout:
    fout["Events"] = arrays
```

Move the result into `nano/tt_dl_powheg_2018_nano_v9_2k.root` inside the build directory from step 1.

## 3. Package and publish the release

```bash
cd build
tar czf ../ci-testdata-v1.tar.gz jsonpog nano
cd ..

gh release create ci-testdata-v1 ci-testdata-v1.tar.gz -R GhentAnalysis/columnflow \
  --title "CI test data v1" \
  --notes "Trimmed tt_dl_powheg 2018 UL NanoAOD (~2k events) + 2018 UL correctionlib JSONs, used by .github/workflows/template_e2e.yaml"
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
