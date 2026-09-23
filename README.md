# NDDMA Performance Modeling

This repository contains the NDDMA performance modeling workflow, including
data collection scripts, fitted model parameters, diagnostic plots, and ad-hoc
prediction tools.

## Functionality

The model is built in four rounds:

| Round | Purpose | Output model |
| --- | --- | --- |
| Round1 | 1D contiguous single/multi-core baseline | `DOC/round1_1d_single_multi_core_model.json` |
| Round2 | 1D non-contiguous GM/UB correction and multi-core `rho` | `DOC/round2_1d_noncontiguous_model.json` |
| Round3 | 2D/3D/4D/5D unified multidimensional expansion | `DOC/round3_multidim_model.json` |
| Round4 | 2D transpose / Round5 `N_G2` UB-contiguous specialization | `DOC/round4_2d_ub_contiguous_ng2_model.json` |

The high-level modeling notes are in:

```text
DOC/nddma_performance_modeling.md
```

## Code Structure

```text
DOC/
  *.md                         Model documents and workflow notes
  *.json                       Fitted model parameters

Modeling/Data_collection/
  common/executables/          Standalone NDDMA profiling harness
  common/tools/                Ad-hoc run and prediction tools
  round1/scripts/              Round1 collect/fit/draw/e2e scripts
  round2/scripts/              Round2 collect/fit/draw/e2e scripts
  round3/scripts/              Round3 collect/fit/draw/e2e scripts
  round4/scripts/              Round4 collect/fit/draw/e2e scripts

ge-develop/
  GE/CANN source tree used by the standalone harness build.
```

Generated analysis outputs are written under `Modeling/Ana/round*/` by default.

## Tools

Predict one arbitrary-dimensional config with the unified Round3 model:

```bash
python3 Modeling/Data_collection/common/tools/predict_nddma_ad_hoc.py \
  12x32x32 512x65536x2097152 1024x32x1 72 int32_t \
  --kernel-repeat 72
```

Predict one 2D transpose / Round4 `N_G2` config:

```bash
python3 Modeling/Data_collection/common/tools/predict_nddma_2d_transpose_ad_hoc.py \
  63x2 1x2 2x1 8 int16_t
```

Run one config through the standalone harness and profiling flow:

```bash
python3 Modeling/Data_collection/common/tools/run_nddma_ad_hoc.py \
  63x2 1x2 2x1 8 int16_t \
  --msprof-bin "$(which msprof)"
```

The prediction tools read parameters from `DOC/*.json`; they do not require
refitting.

## Refit Workflow

Run commands from the repository root. Full collection requires a working Ascend
profiling environment and `msprof`.

Round1:

```bash
python3 Modeling/Data_collection/round1/scripts/e2e.py all \
  --msprof-bin "$(which msprof)"
```

Round2:

```bash
python3 Modeling/Data_collection/round2/scripts/e2e.py all \
  --msprof-bin "$(which msprof)"
```

Round3:

```bash
python3 Modeling/Data_collection/round3/scripts/e2e.py all \
  --msprof-bin "$(which msprof)"
```

Round4:

```bash
python3 Modeling/Data_collection/round4/scripts/e2e.py all \
  --msprof-bin "$(which msprof)"
```

Each round can also be run step by step:

```bash
python3 Modeling/Data_collection/roundX/scripts/e2e.py collect --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/roundX/scripts/e2e.py fit
python3 Modeling/Data_collection/roundX/scripts/e2e.py draw
```

Dependency chain:

```text
Round2 fit inherits Round1 model parameters.
Round3 fit inherits Round2 model parameters.
Round4 fit inherits Round2 model parameters and only fits N_G2.
```

To refit from existing measurement CSVs without recollecting:

```bash
python3 Modeling/Data_collection/round1/scripts/e2e.py fit --measurement-csv /path/to/measurements.csv
python3 Modeling/Data_collection/round2/scripts/e2e.py fit --measurement-csv /path/to/measurements.csv
python3 Modeling/Data_collection/round3/scripts/e2e.py fit --measurement-csv /path/to/measurements.csv
python3 Modeling/Data_collection/round4/scripts/e2e.py fit --measurement-csv /path/to/measurements.csv
```

Use `--dry-run` on any `e2e.py` command to print the invoked subcommands without
executing collection or fitting.
