# NDDMA2 Round3 multidimensional model

Round3 is the multidimensional extension stage. It builds one unified
2D/3D/4D/5D dataset and fits a small scaling layer over the inherited
Round2 one-dimensional non-contiguous model.

Dataset sources:

```text
2D: legacy Round4/Round5 transpose-style 2D dataset
3D: legacy Round6 dataset
4D: legacy Round7 dataset
5D: legacy Round7 dataset
```

Workflow:

```bash
python3 Modeling/round3/e2e.py all --msprof-bin "$(which msprof)"
python3 Modeling/round3/e2e.py collect --msprof-bin "$(which msprof)"
python3 Modeling/round3/e2e.py fit
python3 Modeling/round3/e2e.py draw
```

`fit` requires the Round2 model JSON by default:

```text
Modeling/Ana/round2/round2_1d_noncontiguous_model.json
```

Override it with:

```bash
python3 Modeling/round3/e2e.py fit --round2-model /path/to/round2_1d_noncontiguous_model.json
```

Default outputs:

```text
Modeling/Ana/round3/
├── round3_multidim_factor.csv
├── round3_multidim_model.json
├── round3_multidim_predictions.csv
├── round3_d<dim>_<dtype>_actual_vs_predicted.svg
├── round3_d<dim>_<dtype>_residual.svg
└── collection/
```

Model form:

```text
N_base = round2.base(dtype,total_bytes,block_dim)
T_axis = round2.N_1_correction(dtype,axis_bytes,input_delta,output_delta)
cycles = N_base + d0(dtype,dim) * sum(T_axis)
```

The fitted parameter is only `d0(dtype,dim)`. The inherited one-dimensional
terms are kept visible in `round3_multidim_predictions.csv` as
`n1_terms_json`.
