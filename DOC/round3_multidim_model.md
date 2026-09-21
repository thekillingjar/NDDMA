# NDDMA2 Round3 多维扩展建模

## 目标

Round3 用于 2D/3D/4D/5D 多维 NDDMA 建模。它不重新发明一维非连续公式，而是继承
Round2 的一维 `N_base + N_G + N_GU + rho` 模型，并直接按多维轴展开后求和。
本轮不再拟合额外缩放参数。

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
python3 Modeling/Data_collection/round3/scripts/e2e.py draw
```

拟合依赖 Round2 的 JSON：

```text
Modeling/Ana/round2/round2_1d_noncontiguous_model.json
```

如需指定：

```bash
python3 Modeling/Data_collection/round3/scripts/e2e.py fit \
  --round2-model Modeling/Ana/round2/round2_1d_noncontiguous_model.json
```

## 建模形式

对一个多维样本，先用总数据量计算连续基础项：

```text
N_base = round2.base(dtype,total_bytes,block_dim)
```

再把多维 stride 按 loop 轴拆成若干一维继承修正：

```text
T_axis = round2.N_1_correction(dtype,axis_bytes,input_delta,output_delta)
```

Round3 预测：

```text
cycles = N_base + sum(T_axis)
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
round3_d<dim>_<dtype>_actual_vs_predicted.svg
round3_d<dim>_<dtype>_residual.svg
```
