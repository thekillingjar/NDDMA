# NDDMA2 Round2 一维非连续建模

## 目标

Round2 将原始 NDDMA Round4 的一维非连续模型迁移到 NDDMA2，并补上多核拟合过程。
本目录不依赖旧 `NDDMA` 路径，采集、拟合和绘图入口都在 `NDDMA2/Modeling/Data_collection/round2/scripts`。

## 运行指令

```bash
cd NDDMA2
python3 Modeling/Data_collection/round2/scripts/e2e.py all --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/round2/scripts/e2e.py collect --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/round2/scripts/e2e.py fit
python3 Modeling/Data_collection/round2/scripts/e2e.py draw
```

默认输出目录：

```text
Modeling/Ana/round2
```

## 建模形式

```text
B = bytes_per_core = logical_total_bytes / block_dim
s = min(input_stride*dtype_size, 128)
g = min(1, output_stride-1)
```

连续基础项：

```text
N_base = H_1 + B/T_1, block_dim<=2
N_base = H_2 + B/T_2, block_dim>2
```

先拟合 GM 非连续项：

```text
N_G = (a1+a2*B)*s
```

再拟合 GM+UB 联合非连续项：

```text
N_GU = ((b1+b2*s)+(b3+b4*s)*B)*g
```

最后拟合多核倍率：

```text
rho = (c1+c2*s)+g*(c3+c4*s)
```

最终预测：

```text
N_1 = N_base + N_G + N_GU, block_dim<=2
N_1 = N_base + rho*(N_G+N_GU), block_dim>2
```

## 数据集

当前 factor 总计 6717 行，默认写入：

```text
Modeling/Ana/round2/round2_1d_noncontiguous_factor.csv
```

分组含义：

```text
A: 单核 GM 非连续输入 stride，拟合 N_G
B: 单核 UB 非连续输出 stride，参与 N_GU
C: 单核 GM/UB 联合非连续，拟合 N_GU
F: 多核 base/验证配置
G: 多核 GM 非连续输入校准
H: 多核 GM/UB 联合非连续校准
I/J: 多核验证配置
```

## JSON 解释

`round2_1d_noncontiguous_model.json` 的核心字段：

```text
model
formula
parameters
metrics
```

`formula.fit_order` 记录拟合顺序，固定为：

```text
base -> N_G -> N_GU -> multicore_rho
```

```text
parameters.<dtype>.base
```

连续基础项参数，与任意维多核模型文档一致：

```text
T_1, H_1, T_2, H_2
```

```text
parameters.<dtype>.N_G
```

GM 非连续项参数：

```text
a1, a2
```

```text
parameters.<dtype>.N_GU
```

GM/UB 联合非连续项参数：

```text
b1, b2, b3, b4
```

```text
parameters.<dtype>.rho
```

多核 `block_dim>2` 的倍率参数：

```text
c1, c2, c3, c4
```

## 图解释

`draw` 生成每种 dtype 两张图：

```text
Modeling/Ana/round2/figures/
round2_1d_noncontiguous_actual_vs_predicted_<dtype>.svg
round2_1d_noncontiguous_residual_<dtype>.svg
```

第一类图横轴为 `logical_total_bytes`，纵轴为 cycles，黑点是实测值，蓝色空心点是预测值。

第二类图横轴为 `logical_total_bytes`，纵轴为：

```text
predicted_cycles - actual_cycles
```
