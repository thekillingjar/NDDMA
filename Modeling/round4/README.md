# NDDMA2 Round4 2D transpose multicore model

Round4 is a standalone specialization for the 2D transpose multicore case
called out in the legacy arbitrary-dimensional multicore model.

It is not a generic multidimensional extension. Round3 covers the unified
2D/3D/4D/5D inherited model; Round4 only handles:

```text
output_dims   = [M,N]
input_stride  = [1,is1]
output_stride = [N,1]
block_dim     = 2,4,8,32,64
```

Formula:

```text
T_2d = M*T_1d_multicore + rho_2d*r_singlecore
rho_2d = c1 + c2*s
s = min(is1*dtype_size,128)
```

The full documented form has `c3/c4` output-stride terms, but this dataset
fixes `output_stride=[N,1]`, so `os=1` and `c3/c4` are fixed to zero.

Workflow:

```bash
python3 Modeling/round4/e2e.py all --msprof-bin "$(which msprof)"
python3 Modeling/round4/e2e.py collect --msprof-bin "$(which msprof)"
python3 Modeling/round4/e2e.py fit
python3 Modeling/round4/e2e.py draw
```

Default outputs:

```text
Modeling/Ana/round4/
├── round4_2d_transpose_multicore_factor.csv
├── round4_2d_transpose_multicore_model.json
├── round4_2d_transpose_multicore_predictions.csv
├── round4_2d_transpose_multicore_<dtype>_actual_vs_predicted.svg
├── round4_2d_transpose_multicore_<dtype>_residual.svg
└── collection/
```
