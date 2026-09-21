# NDDMA2 Round1 1D single-core model

This directory contains the NDDMA2 Round1 workflow used as the continuous
base model. Its formula is inherited from the legacy NDDMA Round2 model,
but its data, scripts, and outputs are independent.

The supported interface is:

```bash
python3 e2e.py all --msprof-bin "$(which msprof)"
python3 e2e.py collect --msprof-bin "$(which msprof)"
python3 e2e.py fit
python3 e2e.py draw
```

The fitted scope is intentionally narrow:

- `dim=1`
- `block_dim=1`
- contiguous GM and UB
- `enable_store=0`
- `int8_t`, `int16_t`, `int32_t`, `int64_t`

The model is the `block_dim <= 2` branch inherited by Round4:

```text
cycles(dtype, bytes) = alpha(dtype) + bytes / T(dtype)
```

The default output directory is `NDDMA2/Modeling/Ana/round1`:

```text
Ana/round1/
├── round1_1d_single_core_factor.csv
├── round1_1d_single_core_model.json
├── round1_1d_single_core_predictions.csv
├── round1_1d_single_core_<dtype>.svg
└── collection/
    ├── build/
    ├── profiling_raw/
    ├── experiment/
    └── analysis/
```

The local `Data_collection/common/executables/standalone_nddma` directory
contains the harness and profiling parser. Collection orchestration and
factor generation live directly in `round1/collect.py`, so the workflow
does not depend on a separate Round1 scripts layer.

The harness compiler dependency is expected at `NDDMA2/ge-develop`.
