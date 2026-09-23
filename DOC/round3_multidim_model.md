# NDDMA2 Round3 多维扩展建模

## 目标

Round3 用于 2D/3D/4D/5D 多维 NDDMA 建模。它采用多维统一公式，并完整
继承 Round2 的一维非连续模型参数。Round3 不重新拟合 `N_base`、`N_G`、
`N_GU` 或 `rho`。

## 数据集来源

```text
2D: 旧 NDDMA Round4/Round5 的二维 transpose 类数据规划
3D: 旧 NDDMA Round6 数据规划
4D: 旧 NDDMA Round7 数据规划
5D: 旧 NDDMA Round7 数据规划
```

默认 factor：

```text
Modeling/Ana/round3/round3_multidim_factor.csv
```

默认采集输出：

```text
Modeling/Ana/round3/collection
```

## 运行方式

```bash
cd NDDMA2
python3 Modeling/Data_collection/round3/scripts/e2e.py all --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/round3/scripts/e2e.py collect --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/round3/scripts/e2e.py fit
python3 Modeling/Data_collection/round3/scripts/e2e.py fit \
  --round2-model Modeling/Ana/round2/round2_1d_noncontiguous_model.json
python3 Modeling/Data_collection/round3/scripts/e2e.py draw
```

Round3 默认读取：

```text
Modeling/Ana/round2/round2_1d_noncontiguous_model.json
```

Round2 JSON 中的 `base`、`N_G`、`N_GU` 和 `rho` 参数全部用于
Round3 的多维展开。

## 建模形式

对一个多维样本，先反转 API 维度顺序，得到 kernel 逻辑 loop 顺序：

```text
output_dim    = [ls0, ls1, ..., ls{D-1}]
input_stride  = [is0, is1, ..., is{D-1}]
output_stride = [os0, os1, ..., os{D-1}]
```

总数据量：

```text
B = product(ls_j) * dtype_size
```

连续基础项只计算一次，使用 Round2 的一维多核基础项。先定义：

```text
B_core = B / block_dim
N_base = B_core*(block_dim/T_1+h_1)+H_1, block_dim<=2
N_base = B_core*(block_dim/T_2+h_2)+H_2, block_dim>2
```

第 `j` 维修正项使用去掉更内侧 `0..j-1` 维后的数据量：

```text
B_j = B / product_{t=0}^{j-1}(ls_t)
B_0 = B
```

等效 stride：

```text
is_hat_0 = is0
os_hat_0 = os0

is_hat_j = abs(is_j - sum_{t=0}^{j-1}(ls_t*is_t)) + 1, j>=1
os_hat_j = abs(os_j - sum_{t=0}^{j-1}(ls_t*os_t)) + 1, j>=1
```

每一维使用 Round2 JSON 中的一维多核修正项：

```text
T_axis_j = N_1_prime(dtype,B_j,is_hat_j,os_hat_j,block_dim)
```

其中：

```text
N_1_prime = N_G + N_GU, block_dim<=2
N_1_prime = rho*(N_G + N_GU), block_dim>2
```

Round3 最终预测：

```text
N_D = N_base + sum_j(T_axis_j)
```

因此 JSON 中可以清楚区分：

```text
Round2 继承参数: base, N_G, N_GU, rho
Round3 新拟合参数: none
```

## 输出文件

```text
round3_multidim_model.json
```

包含：

```text
formula
fit_scope
metrics
```

```text
round3_multidim_predictions.csv
```

逐点记录：

```text
actual_cycles
n_base_cycles
t1_cycles
predicted_cycles
error_cycles
n1_terms_json
```

其中 `n1_terms_json` 是每个维度轴展开后的一维修正项明细。

绘图输出：

```text
Modeling/Ana/round3/figures/
round3_d<dim>_<dtype>_residual.svg
```

`round3_d<dim>_<dtype>_residual.svg` 横轴为 `data_volume_bytes`，纵轴为：

```text
(predicted_cycles - actual_cycles) / actual_cycles
```

不同 `layout_pattern` 场景使用不同颜色标注。
