# NDDMA2 Round4 二维转置多核特化建模

## 定位

Round4 是独立的二维转置多核建模，用于提高任意维度统一模型在二维转置场景下的准确度。

适用配置：

```text
output_dims   = [M,N]
input_stride  = [1,is1]
output_stride = [N,1]
block_dim     = [2,4,8,32,64]
```

## 公式

```text
T_2d = M*T_1d_multicore + rho_2d*r_singlecore
```

其中：

```text
rho_2d = (c1+c2*s)+min(1,os-1)*(c3+c4*s)
s = min(is1*dtype_size,128)
```

当前 O 组数据固定：

```text
output_stride = [N,1]
```

因此 `os=1`，只能识别：

```text
rho_2d = c1+c2*s
```

`c3/c4` 在 JSON 中保留但固定为 0。

单核二维 residual 继承旧模型：

```text
r_singlecore=(a0+a1*S)*N*(M-1)+b*M+d
S=is1*dtype_size
```

## 运行

```bash
cd NDDMA2
python3 Modeling/round4/e2e.py all --msprof-bin "$(which msprof)"
python3 Modeling/round4/e2e.py collect --msprof-bin "$(which msprof)"
python3 Modeling/round4/e2e.py fit
python3 Modeling/round4/e2e.py draw
```

默认输出：

```text
Modeling/Ana/round4
```

## 输出解释

```text
round4_2d_transpose_multicore_model.json
```

包含：

```text
formula
fit_scope
inherited_parameters
dtype_models
metrics
```

`dtype_models.<dtype>` 中：

```text
c1, c2: 当前数据可识别的 rho 参数
c3, c4: 当前固定为 0
```

```text
round4_2d_transpose_multicore_predictions.csv
```

逐点记录：

```text
baseline_cycles = M*T_1d_multicore
single_core_residual_cycles = r_singlecore
rho_observed
rho_predicted
predicted_cycles
error_cycles
```
