# Round2 1D single-core model

This directory contains the reduced Round2 workflow used as the Round4
continuous-base model. It is isolated from the legacy
`NDDMA/Modeling/Ana/round2` implementation.

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

`fit` writes `generated/round2_1d_single_core_model.json`. The collection
The local `Data_collection/common/executables/standalone_nddma` directory
contains the harness and profiling parser. Collection orchestration and
factor generation live directly in `round2/collect.py`, so the workflow
does not depend on a separate Round2 scripts layer.
