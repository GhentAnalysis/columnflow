# 04 — Awkward Array Reference for Columnflow

Columnflow event data is stored as `ak.Array` named `events`. Fields like `events.Jet` are collections (variable-length sub-arrays); scalar fields like `events.MET.pt` are uniform arrays over events.

Always import awkward via `maybe_import`:
```python
from columnflow.util import maybe_import
ak = maybe_import("awkward")
np = maybe_import("numpy")
```

---

## Accessing Fields

```python
# Scalar per-event field
events.event                       # event number, shape (N,)
events.MET.pt                      # MET pT, shape (N,)

# Collection field — variable-length inner dimension
events.Jet.pt                      # shape (N, var)
events.Electron.eta                # shape (N, var)

# Brace expansion works in uses/produces but NOT in ak access — use dot notation
events.Jet.pt                      # correct
events["Jet", "pt"]                # also valid
```

---

## Indexing and Slicing

```python
# First jet in every event (raises if any event has 0 jets)
events.Jet.pt[:, 0]

# Safe nth-object access with fill for missing
from columnflow.columnar_util import Route, EMPTY_FLOAT
jet3_pt = Route("Jet.pt[:, 2]").apply(events, null_value=EMPTY_FLOAT)

# Select specific events
selected_events = events[event_mask]   # event_mask is bool, shape (N,)

# Select objects within events using an index array
jet_indices = ak.local_index(events.Jet.pt)[jet_mask]
selected_jets = events.Jet[jet_indices]
```

---

## Boolean Masks

```python
# Object-level mask (same shape as the collection)
jet_mask = (events.Jet.pt > 25.0) & (abs(events.Jet.eta) < 2.4)  # shape (N, var)

# Event-level mask derived from object counts
event_mask = ak.sum(jet_mask, axis=1) >= 2                         # shape (N,)

# Multi-condition
mask = (
    (events.Electron.pt > 15.0) &
    (abs(events.Electron.eta) < 2.5) &
    (events.Electron.cutBased >= 3)
)

# Invert
not_mask = ~mask
```

---

## Reduction Operations (axis=1 for per-event sums)

```python
# Sum over objects in each event
ht = ak.sum(events.Jet.pt, axis=1)                    # shape (N,)
n_jet = ak.sum(events.Jet.pt > 0, axis=1)             # count jets (bool sum)
n_bjet = ak.sum(events.Jet.btagDeepFlavB > 0.5, axis=1)

# Min / max per event
max_jet_pt = ak.max(events.Jet.pt, axis=1)
min_jet_eta = ak.min(abs(events.Jet.eta), axis=1)

# Any / all per event
has_forward_jet = ak.any(abs(events.Jet.eta) > 2.4, axis=1)
all_jets_central = ak.all(abs(events.Jet.eta) < 2.4, axis=1)

# Mean per event
mean_jet_pt = ak.mean(events.Jet.pt, axis=1)

# axis=0 for global sums (used in stats)
total_mc_weight = ak.sum(events.mc_weight)
```

---

## Padding and Filling None

```python
# Pad to at least N objects, filling with None
padded = ak.pad_none(events.Jet, 4, axis=1)           # shape (N, >=4) with None

# Fill None with scalar
filled = ak.fill_none(padded.pt, EMPTY_FLOAT)

# Fill None with dict (for record arrays)
fill_values = {"pt": -999.0, "eta": -999.0, "phi": -999.0, "mass": -999.0}
filled = ak.fill_none(ak.pad_none(events.Lepton, 2, axis=1), fill_values)

# Access padded element safely
lepton0_pt = ak.fill_none(ak.pad_none(events.Lepton.pt, 1, axis=1)[:, 0], EMPTY_FLOAT)

# Check for None
is_missing = ak.is_none(events.Jet.pt, axis=1)
```

---

## Sorting and Argsort

```python
# Sort jets by pT descending
sorted_jets = events.Jet[ak.argsort(events.Jet.pt, axis=1, ascending=False)]

# Sort and take first N
leading_2_jets = events.Jet[ak.argsort(events.Jet.pt, axis=1, ascending=False)][:, :2]
```

---

## Concatenation and Combination

```python
# Merge two collections (e.g. electrons and muons into leptons)
leptons = ak.concatenate([events.Electron, events.Muon], axis=1)
# Sort by pt descending
leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]

# Zip fields into a record array
dijet = ak.zip({
    "pt": events.Jet.pt[:, 0] + events.Jet.pt[:, 1],
    "eta": events.Jet.eta[:, 0],
})
```

---

## Flattening

```python
# Flatten all jets from all events into a 1D array (for histogram filling)
all_jet_pts = ak.flatten(events.Jet.pt)       # shape (sum_of_jet_counts,)

# Flatten with axis=None (full recursive flatten)
flat = ak.flatten(events.Jet.pt, axis=None)
```

---

## Type Conversions

```python
# Convert to numpy (only works on regular arrays — use after flatten/reduction)
arr_np = ak.to_numpy(ak.sum(events.Jet.pt, axis=1))

# Cast types
events_int = ak.values_astype(events.Jet.nConstituents, np.int32)
events_float = ak.values_astype(events.Jet.pt, np.float32)
```

---

## Conditional Selection (where)

```python
# Per-event conditional
selected_pt = ak.where(event_mask, events.Jet.pt[:, 0], EMPTY_FLOAT)

# Per-object conditional
corrected_pt = ak.where(events.Jet.pt > 50, events.Jet.pt * 1.02, events.Jet.pt)
```

---

## local_index — Creating Index Arrays for SelectionResult.objects

The standard way to create index arrays for `SelectionResult.objects`:

```python
# Create indices of jets that pass the selection
jet_mask = (events.Jet.pt > 25.0) & (abs(events.Jet.eta) < 2.4)
jet_indices = ak.local_index(events.Jet.pt, axis=1)[jet_mask]

# Use in SelectionResult
return events, SelectionResult(
    objects={"Jet": {"Jet": jet_indices}},
)
```

---

## set_ak_column — Writing Columns

Never mutate `events` fields directly. Always use `set_ak_column`:

```python
from columnflow.columnar_util import set_ak_column

# Add a new scalar column
events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1), value_type=np.float32)

# Add a nested column (inside an existing collection)
events = set_ak_column(events, "Jet.pt_corr", events.Jet.pt * 1.02)

# Replace an existing collection field
events = set_ak_column(events, "Jet.btagDeepFlavB", corrected_btag)
```

---

## Lorentz Vectors (coffea behaviour)

After attaching coffea behaviour, awkward records expose four-vector methods:

```python
from columnflow.production.util import attach_coffea_behavior

events = self[attach_coffea_behavior](events, **kwargs)

# Four-momentum operations (only after attach_coffea_behavior)
jet1 = events.Jet[:, 0]
jet2 = events.Jet[:, 1]
dijet_mass = (jet1 + jet2).mass
delta_r = jet1.delta_r(jet2)
dijet_pt = (jet1 + jet2).pt
```

To build a Lorentz vector manually:

```python
# Inside a function (NOT at module level)
import coffea.nanoevents.methods.vector

vec = ak.zip(
    {"pt": arr.pt, "eta": arr.eta, "phi": arr.phi, "mass": arr.mass},
    with_name="PtEtaPhiMLorentzVector",
    behavior=coffea.nanoevents.methods.vector.behavior,
)
mass = (vec[:, 0] + vec[:, 1]).mass
```

---

## Common Patterns in Columnflow

### Count objects after selection

```python
jet_mask = (events.Jet.pt > 25) & (abs(events.Jet.eta) < 2.4)
n_jet = ak.sum(jet_mask, axis=1)   # per-event count
events = set_ak_column(events, "n_jet", n_jet)
```

### Per-process event weights

```python
# Sum of MC weights per process (used in stats)
for pid in np.unique(ak.to_numpy(events.process_id)):
    process_mask = events.process_id == pid
    stats[f"sum_mc_weight_per_process"][int(pid)] += ak.sum(events.mc_weight[process_mask])
```

### HT (scalar sum of jet pT)

```python
ht = ak.sum(events.Jet.pt, axis=1)
events = set_ak_column(events, "ht", ht, value_type=np.float32)
```

### Leading lepton pT

```python
lep_pt = ak.fill_none(ak.pad_none(events.Lepton.pt, 1, axis=1)[:, 0], EMPTY_FLOAT)
events = set_ak_column(events, "lep1_pt", lep_pt, value_type=np.float32)
```

### Invariant mass of two objects

```python
# Requires coffea behaviour to be attached
mll = (events.Lepton[:, 0] + events.Lepton[:, 1]).mass
z_veto = abs(mll - 91.2) < 15.0
```

---

## Gotchas

- `axis=1` operates over the **object** (inner) dimension; `axis=0` operates over **events** (outer).
- Arithmetic between collections with different lengths requires `ak.broadcast_arrays` or `ak.zip`.
- Never iterate over `ak.Array` events in Python — always use vectorized operations.
- `ak.Array` returned from a TAF is a copy (with shared underlying data buffers), not the same object as the input; this is intentional and efficient.
- `ak.to_numpy()` fails on variable-length (jagged) arrays — flatten first.
