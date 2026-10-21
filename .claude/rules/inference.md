---
paths:
  - "**/inference/**"
---

# InferenceModel Rules

## Decorator and structure

```python
from columnflow.inference import inference_model, ParameterType, ParameterTransformation


@inference_model
def default(self):

    # --- Categories (one per distribution used in the fit) ---
    self.add_category(
        "sr",
        config_category="sr",           # name of order Category in config
        config_variable="jet1_pt",      # variable whose histogram to use
        config_data_datasets=["data_mu_b", "data_mu_c"],
        mc_stats=True,                   # add MC stat uncertainty (Barlow-Beeston)
    )

    # --- Processes ---
    self.add_process(
        "TT",
        is_signal=False,
        config_process="tt",
        config_mc_datasets=["tt_dl_powheg"],
    )
    self.add_process(
        "ST",
        is_signal=True,
        config_process="st",
    )

    # --- Parameters (nuisances) ---

    # Rate uncertainty (log-normal)
    lumi = self.config_inst.x.luminosity
    for unc_name in lumi.uncertainties:
        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
            transformations=[ParameterTransformation.symmetrize],
        )

    # Weight-based shape uncertainty
    self.add_parameter(
        "mu",
        process=["TT", "ST"],
        type=ParameterType.shape,
        config_shift_source="mu",   # links to shifts mu_up / mu_down in config
    )

    # Selection-modifying shape uncertainty (JEC)
    self.add_parameter(
        "jec",
        process=["TT", "ST"],
        type=ParameterType.shape,
        config_shift_source="jec",
    )

    # Dedicated-dataset uncertainty
    self.add_parameter(
        "tune",
        process="TT",
        type=ParameterType.shape,
        config_shift_source="tune",
    )
```

## Key rules

- `config_shift_source` must match the shift name prefix in the config (e.g. `"mu"` → shifts `mu_up` / `mu_down`).
- Register in `law.cfg` under `inference_modules`.
- Run with `law run cf.CreateDatacards --inference-model default --variables jet1_pt`.
