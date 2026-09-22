# NDDMA2 Round4 Round5 N_G2 二维 UB 连续特化建模

## 定位

Round4 采用原始 NDDMA Round5 的 `N_G2` 二维 UB 连续特化模型，用于
`[M,N]/[is2,is1]/[N,1]` 场景。

适用配置：

```text
output_dims   = [M,N]
input_stride  = [is2,is1]
output_stride = [N,1]
is2 < is1
block_dim     = [2,4,8,32,64]
```

## 公式

```text
B  = M*N*dtype_size
B1 = N*dtype_size

N_base = round2.base(dtype,B,k)
N_G1   = round2.N_G(dtype,B1,is1,k)
N_G2   = (g10+g11_M*M)*is2 + g00 + g01_M*M

N_2 = N_base + N_G1*N_G2
```

其中 `N_base`、`N_G1` 从 Round2 一维非连续模型 JSON 继承；Round4
只拟合二维外层乘子 `N_G2` 的四个参数。继承的 `N_base` 使用当前
Round1/Round2 base 形式：

```text
N_base = B*(block_dim/T_1+h_1)+H_1, block_dim<=2
N_base = B*(block_dim/T_2+h_2)+H_2, block_dim>2
```

## 运行

```bash
cd NDDMA2
python3 Modeling/Data_collection/round4/scripts/e2e.py all --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/round4/scripts/e2e.py collect --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/round4/scripts/e2e.py fit
python3 Modeling/Data_collection/round4/scripts/e2e.py draw
```

默认继承的 Round2 模型：

```text
Modeling/Ana/round2/round2_1d_noncontiguous_model.json
```

如需指定其他 Round2 模型：

```bash
python3 Modeling/Data_collection/round4/scripts/e2e.py fit \
  --round2-model /path/to/round2_1d_noncontiguous_model.json
```

默认输出：

```text
Modeling/Ana/round4
```

## 输出解释

```text
round4_2d_ub_contiguous_ng2_model.json
```

包含：

```text
model
formula
parameters
metrics
```

`parameters.N_G2.<dtype>` 中：

```text
g10
g11_M
g00
g01_M
```

`round2_model_source` 记录实际继承的 Round2 JSON 路径。

`parameters.one_dimensional` 保存该二维特化公式继承的一维基础项、
GM 非连续修正项参数和 `rho` 参数；这些参数不在 Round4 中重新拟合。

```text
round4_2d_ub_contiguous_ng2_predictions.csv
```

逐点记录：

```text
n_base_cycles
n_g1_cycles
observed_n_g2
predicted_n_g2
actual_cycles
predicted_cycles
error_cycles
```

绘图输出：

```text
Modeling/Ana/round4/figures/
round4_2d_ub_contiguous_ng2_<dtype>_actual_vs_predicted.svg
round4_2d_ub_contiguous_ng2_<dtype>_residual.svg
round4_2d_ub_contiguous_ng2_<dtype>_observed_ng2_vs_m.svg
round4_2d_ub_contiguous_ng2_<dtype>_error_vs_bytes.svg
```

`observed_ng2_vs_m` 横轴为 `M`，纵轴为：

```text
(N2 - N_base) / N_G1
```

不同外层 `is2` 使用不同颜色；同一 `is2/M` 下不同 `N/block_dim/is1`
的原始点保留，连线使用这些点的中位数。

`error_vs_bytes` 横轴为 `bytes_per_core`，纵轴为
`predicted_cycles - actual_cycles`，不同 `block_dim` 使用不同颜色。
