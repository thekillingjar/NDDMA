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

N_base = N_base(B,k)
N_G1   = N_1'(B1,is1,1,k)
N_G2   = (g10+g11_M*M)*is2 + g00 + g01_M*M

N_2 = N_base + N_G1*N_G2
```

其中 `N_base`、`N_G1` 继承任意维统一一维项；Round4 只拟合二维外层
乘子 `N_G2` 的四个参数。

## 运行

```bash
cd NDDMA2
python3 Modeling/Data_collection/round4/scripts/e2e.py all --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/round4/scripts/e2e.py collect --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/round4/scripts/e2e.py fit
python3 Modeling/Data_collection/round4/scripts/e2e.py draw
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

`parameters.one_dimensional` 保存该二维特化公式继承的一维基础项和
GM 非连续修正项参数。

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
```
