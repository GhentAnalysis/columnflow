---
paths:
  - "**/categorization/**"
---

# Categorizer Rules

## Decorator and return signature

```python
from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import
ak = maybe_import("awkward")

@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # Returns (events, boolean_mask_1d) — True = event belongs to this category
    return events, ak.ones_like(events.event, dtype=bool)

@categorizer(uses={"n_jet", "n_bjet"})
def cat_sr(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, (events.n_jet >= 4) & (events.n_bjet >= 2)
```

## Registering in config

```python
from columnflow.config_util import add_category

add_category(cfg, name="incl",  id=1, selection="cat_incl",  label="Inclusive")
add_category(cfg, name="sr",    id=2, selection="cat_sr",    label="Signal Region")
add_category(cfg, name="cr_1b", id=3, selection="cat_cr_1b", label="CR (1b)")
```

The `selection` argument is the **name of the Categorizer function** (a string), not a boolean expression.

## Key rules

- Always include `"event"` in `uses` for inclusive categorizers (it's always available).
- Categorizers run inside `CreateHistograms` after `ProduceColumns`; they read columns produced by Producers.
- Categories are mutually exclusive by convention; combine them via the `--categories` CLI flag to run several in parallel.
- Register the file in `law.cfg` under `categorization_modules`.
