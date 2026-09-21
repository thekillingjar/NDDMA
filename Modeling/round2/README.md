# NDDMA2 Round2 1D non-contiguous model

Round2 migrates the legacy NDDMA Round4 one-dimensional non-contiguous model
into NDDMA2. It is independent from the old `NDDMA` directory and uses the
local standalone harness under:

```text
NDDMA2/Modeling/Data_collection/common/executables/standalone_nddma
```

The workflow interface is:

```bash
python3 Modeling/round2/e2e.py all --msprof-bin "$(which msprof)"
python3 Modeling/round2/e2e.py collect --msprof-bin "$(which msprof)"
python3 Modeling/round2/e2e.py fit
python3 Modeling/round2/e2e.py draw
```

Default outputs go under:

```text
NDDMA2/Modeling/Ana/round2/
├── round2_1d_noncontiguous_factor.csv
├── round2_1d_noncontiguous_model.json
├── round2_1d_noncontiguous_predictions.csv
├── figures/
│   ├── round2_1d_noncontiguous_actual_vs_predicted_<dtype>.svg
│   └── round2_1d_noncontiguous_residual_<dtype>.svg
└── collection/
```

The model keeps the original staged order:

```text
N_base = B/T_le2 + alpha_le2, k<=2
N_base = B/T_gt2 + alpha_gt2, k>2
s = min(input_stride*dtype_size, 128)
N_G = (a1+a2*B)*s
N_GU = ((b1+b2*s)+(b3+b4*s)*B)*min(1, output_stride-1)
rho = (c1+c2*s)+min(1, output_stride-1)*(c3+c4*s)

N_1 = N_base + N_G + N_GU, k<=2
N_1 = N_base + rho*(N_G+N_GU), k>2
```

Here `B=bytes_per_core=logical_total_bytes/block_dim`.

Fitting order:

```text
base -> N_G -> N_GU -> multicore rho
```

Groups:

```text
A: single-core input stride, fits N_G
B: single-core output stride, contributes to N_GU
C: single-core input/output joint stride, fits N_GU
F: multi-core base/validation rows
G: multi-core input stride calibration
H: multi-core input/output joint calibration
I/J: multi-core validation rows
```
