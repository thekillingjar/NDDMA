# NDDMA2 Round1 一维单核与多核连续建模

## 1. 目标

NDDMA2 Round1 建立多核一维连续 NDDMA 的基础模型

## 2. 目录和运行入口

Round1 脚本：

```text
NDDMA2/Modeling/round1/
├── e2e.py
├── collect.py
├── fit.py
├── draw.py
└── README.md
```

本地 harness：

```text
NDDMA2/Modeling/Data_collection/common/executables/standalone_nddma/
```

编译依赖：

```text
NDDMA2/ge-develop/
```

默认产物：

```text
NDDMA2/Modeling/Ana/round1/
├── round1_1d_single_core_factor.csv
├── round1_1d_single_multi_core_model.json
├── round1_1d_single_core_predictions.csv
├── figures/
│   └── round1_1d_single_core_<dtype>.svg
└── collection/
    ├── build/
    ├── profiling_raw/
    ├── experiment/
    └── analysis/
```

## 3. 建模范围

每条配置固定为：

```text
dim = 1
input_stride = 1
output_stride = 1
GM = contiguous
UB = contiguous
enable_store = 0
src_offset_elem = 0
dst_offset_elem = 0
```

`block_dim` 是多核变量，取：

```text
block_dim = 1, 2, 3, ..., 56
```

覆盖四种 dtype：

```text
int8_t
int16_t
int32_t
int64_t
```

## 4. 数据集

### 4.1 每种 dtype 的 bytes 档位

| dtype | `bytes_per_core` 档位 | `block_dim` | 配置数 |
| --- | --- | --- | ---: |
| `int64_t` | 4096, 8192, 16384, 32768, 49152, 61440, 90112, 130048, 180224, 256000 | 1..56 | 560 |
| `int32_t` | 4096, 8192, 16384, 32768, 49152, 61440, 90112, 130048, 180224, 256000 | 1..56 | 560 |
| `int16_t` | 4096, 8192, 16384, 32768, 49152, 61440, 90112, 130048 | 1..56 | 448 |
| `int8_t` | 4096, 8192, 16384, 32768, 49152, 61440 | 1..56 | 336 |

总配置数：

```text
560 + 560 + 448 + 336 = 1904
```

### 4.2 `output_dims`

`output_dims` 是单核 payload 的一维长度：

```text
output_dims = bytes_per_core / dtype_size
```

例如：

```text
int32_t, bytes_per_core=16384
output_dims=4096
block_dim=8
logical_total_bytes=131072
```

### 4.3 字节量定义

当前模型必须区分两个字节量：

```text
bytes_per_core       = 单个核负责的搬运量
logical_total_bytes  = 所有核的逻辑总搬运量
                      = bytes_per_core * block_dim
```

拟合公式中的自变量是：

```text
bytes_per_core
```

不是 `logical_total_bytes`。这样才能比较不同核数下每个核的连续搬运开销。

### 4.4 Factor 文件

默认 factor 文件：

```text
NDDMA2/Modeling/Ana/round1/round1_1d_single_core_factor.csv
```

虽然文件名保留了 `single_core` 历史名称，但当前数据已包含
`block_dim=1..56`，模型是单核和多核连续模型。

token 形式：

```text
r1_1d_single_core_<dtype>_b<bytes_per_core>_c<block_dim>
```

例如：

```text
r1_1d_single_core_int32_t_b16384_c8
```

## 5. 建模公式

### 5.1 每个 dtype、每个核数的局部模型

首先对每个 `(dtype, block_dim)` 独立拟合：

```text
cycles(dtype, block_dim, B)
    = alpha(dtype, block_dim)
    + B / T(dtype, block_dim)
```

其中：

```text
B = bytes_per_core
```

等价地：

```text
cycles = alpha + cycles_per_byte * B
cycles_per_byte = 1 / T
```

每个 `(dtype, block_dim)` 有至少两个 bytes 档位，因此可以拟合截距和斜率。

### 5.2 分段模型

Round1 最终把 `block_dim` 分为两个分支：

```text
block_dim <= 2:
    cycles = alpha_le2(dtype) + bytes_per_core / T_le2(dtype)

block_dim > 2:
    cycles = alpha_gt2(dtype) + bytes_per_core / T_gt2(dtype)
```

对应关系：

```text
le2: block_dim = 1, 2
gt2: block_dim = 3..56
```

### 5.3 分支参数的取得方式

对于某个 dtype 和某个分支：

1. 对分支内每个 `block_dim` 单独拟合 `alpha` 和 `T`。
2. 对所有局部拟合结果的 `alpha` 取算术平均。
3. 对所有局部拟合结果的 `T` 取算术平均。
4. 使用这两个平均参数预测该分支的全部样本。

注意：代码是分别平均 `alpha` 和 `T`，不是先平均
`cycles_per_byte` 再求倒数。

### 5.4 参数单位

```text
alpha             : cycles
T_bytes_per_cycle : bytes/cycle
cycles_per_byte   : cycles/byte
```

预测时推荐使用：

```python
predicted = alpha + bytes_per_core / T_bytes_per_cycle
```


## 6. 运行命令

从 `NDDMA2` 根目录执行。

### 6.1 完整流程

```bash
python3 Modeling/round1/e2e.py all \
  --msprof-bin "$(which msprof)"
```

执行顺序：

```text
collect -> fit -> draw
```

### 6.2 生成、编译和采集

```bash
python3 Modeling/round1/e2e.py collect \
  --msprof-bin "$(which msprof)"
```

该步骤会：

1. 生成 1904 条 factor。
2. 编译本地 `standalone_nddma` harness。
3. 执行 NDDMA。
4. 执行 `msprof`。
5. 解析 `op_summary` 并输出 profiling 参数 CSV。

采集目录：

```text
NDDMA2/Modeling/Ana/round1/collection/
├── build/
├── profiling_raw/
├── experiment/
├── analysis/
├── command.log
└── analysis.log
```

`experiment/` 会在运行 harness 前创建，因为 harness 会直接打开：

```text
experiment/experiment.log
```

### 6.3 只拟合

```bash
python3 Modeling/round1/e2e.py fit
```

默认搜索：

```text
NDDMA2/Modeling/Ana/round1/collection/
```

测量文件搜索顺序：

1. `collection/measurements.csv`
2. 递归搜索 `profiling_with_params_mean.csv`
3. 递归搜索 `profiling_with_params.csv`

也可以显式指定：

```bash
python3 Modeling/round1/e2e.py fit \
  --measurement-csv /path/to/measurements.csv
```

### 6.4 只绘图

```bash
python3 Modeling/round1/e2e.py draw
```

绘图读取：

```text
NDDMA2/Modeling/Ana/round1/round1_1d_single_core_predictions.csv
```

### 6.5 自定义路径

```bash
python3 Modeling/round1/e2e.py all \
  --msprof-bin "$(which msprof)" \
  --run-output-dir /path/to/collection \
  --model-output-dir /path/to/model
```

## 8. 拟合 JSON

输出文件：

```text
NDDMA2/Modeling/Ana/round1/round1_1d_single_multi_core_model.json
```

文件名沿用历史名称，但 JSON 模型标识已经反映当前同时覆盖单核和多核：

```json
"model": "NDDMA_ROUND1_1D_PIECEWISE_SINGLE_MULTI_CORE"
```

### 8.1 顶层结构

```json
{
  "model": "...",
  "formula": {},
  "fit_scope": {},
  "dtype_models": {}
}
```

### 8.2 `formula`

示例：

```json
"formula": {
  "name": "round2_c_group_piecewise_constant_t_alpha",
  "split_block_dim": 2,
  "le2": "cycles = alpha_le2(dtype) + bytes_per_core / T_le2(dtype)",
  "gt2": "cycles = alpha_gt2(dtype) + bytes_per_core / T_gt2(dtype)",
  "bytes_definition": "bytes_per_core = logical_total_bytes / block_dim",
  "fitting_method": "..."
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `name` | 模型方法名称 |
| `split_block_dim` | 分支边界，本轮为 2 |
| `le2` | 单核/双核分支公式 |
| `gt2` | 多核分支公式 |
| `bytes_definition` | 自变量定义 |
| `fitting_method` | 局部拟合和分支平均方法 |

### 8.3 `fit_scope`

示例：

```json
"fit_scope": {
  "dim": 1,
  "block_dim_values": [1, 2, 3, "...", 56],
  "input_stride": 1,
  "output_stride": 1,
  "layout": "contiguous",
  "dtype_order": ["int8_t", "int16_t", "int32_t", "int64_t"],
  "sample_count": 1904,
  "sample_count_by_dtype": {
    "int8_t": 336,
    "int16_t": 448,
    "int32_t": 560,
    "int64_t": 560
  }
}
```

`sample_count` 是重复聚合后的样本数。完整 C 组数据应为：

```text
int8_t  = 6  × 56 = 336
int16_t = 8  × 56 = 448
int32_t = 10 × 56 = 560
int64_t = 10 × 56 = 560
total   = 1904
```

### 8.4 `dtype_models`

每种 dtype 都包含：

```json
"int32_t": {
  "sample_count": 560,
  "block_dim_min": 1,
  "block_dim_max": 56,
  "bytes_per_core_min": 4096,
  "bytes_per_core_max": 256000,
  "branches": {
    "le2": {},
    "gt2": {}
  },
  "metrics": {}
}
```

### 8.5 分支参数

`branches.le2` 和 `branches.gt2` 的结构为：

```json
{
  "name": "gt2",
  "block_dims": [3, 4, 5],
  "alpha": 0.0,
  "T_bytes_per_cycle": 0.0,
  "cycles_per_byte": 0.0,
  "per_block_fits": [],
  "metrics": {}
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `name` | 分支名：`le2` 或 `gt2` |
| `block_dims` | 参与该分支平均的核数 |
| `alpha` | 分支平均固定开销，单位 cycles |
| `T_bytes_per_cycle` | 分支平均搬运速率，单位 bytes/cycle |
| `cycles_per_byte` | `1 / T_bytes_per_cycle` |
| `per_block_fits` | 每个核数的局部线性拟合参数 |
| `metrics` | 用分支平均参数回代全部分支样本后的误差 |

`per_block_fits` 中每项对应一个 `(dtype, block_dim)`：

```json
{
  "block_dim": 8,
  "alpha": 0.0,
  "T_bytes_per_cycle": 0.0,
  "cycles_per_byte": 0.0,
  "sample_count": 10
}
```

它用于诊断不同核数的局部参数变化；最终预测使用分支平均参数，
而不是直接使用某个核数的局部参数。

### 8.6 误差指标

每个 dtype 和每个分支都记录：

```json
"metrics": {
  "count": 0,
  "rmse_cycles": 0.0,
  "mae_cycles": 0.0,
  "max_absolute_error_cycles": 0.0
}
```

定义：

```text
error = predicted - actual
RMSE  = sqrt(mean(error^2))
MAE   = mean(abs(error))
MaxAE = max(abs(error))
```

## 9. Predictions CSV

输出：

```text
NDDMA2/Modeling/Ana/round1/round1_1d_single_core_predictions.csv
```

字段：

| 字段 | 含义 |
| --- | --- |
| `token` | 配置标识 |
| `dtype` | 数据类型 |
| `block_dim` | 核数 |
| `bytes_per_core` | 模型自变量 |
| `logical_total_bytes` | 逻辑总搬运量 |
| `branch` | `le2` 或 `gt2` |
| `actual_cycles` | 聚合后的实际 cycles |
| `predicted_cycles` | 分段模型预测值 |
| `error_cycles` | `predicted - actual` |

该文件既用于绘图，也用于逐点检查模型误差。

## 10. 图的解释

### 10.1 图文件

四种 dtype 各生成一张：

```text
NDDMA2/Modeling/Ana/round1/figures/
round1_1d_single_core_int8_t.svg
round1_1d_single_core_int16_t.svg
round1_1d_single_core_int32_t.svg
round1_1d_single_core_int64_t.svg
```

### 10.2 坐标轴

横轴：

```text
logical total bytes
```

纵轴：

```text
cycles
```

横轴选择逻辑总字节数，是为了在同一张图上观察不同 `block_dim`
下的完整搬运规模。实际模型计算仍使用 `bytes_per_core`。

### 10.3 图中元素

- 黑色实心圆：`actual_cycles`
- 蓝色空心圆：`predicted_cycles`

预测点不是一条连续折线，因为不同核数使用不同的 `bytes_per_core`
和分支参数；用散点可以避免把不同核数的预测错误连接起来。

### 10.4 判断方法

理想情况：

- 蓝色预测点接近黑色实际点
- 单核/双核点与多核点都没有明显系统性偏差
- 随逻辑总字节数增加，没有单向发散

常见异常：

| 图形现象 | 可能含义 |
| --- | --- |
| `block_dim<=2` 点整体偏离 | `le2` 分支参数不适合单核/双核 |
| `block_dim>2` 点整体偏离 | `gt2` 分支参数不适合多核 |
| 大字节量误差变大 | `T` 不准确或线性假设不足 |
| 小字节量误差大、大字节量较好 | `alpha` 不准确 |
| 某些核数单独异常 | 该核数存在额外调度或硬件行为 |
| 两个分支都呈明显弯曲 | 需要进一步引入连续的 `T(block_dim)` 或 `alpha(block_dim)` 模型 |

图像只提供直观诊断，正式误差应查看 JSON 的：

```text
dtype_models.<dtype>.branches.le2.metrics
dtype_models.<dtype>.branches.gt2.metrics
dtype_models.<dtype>.metrics
```

## 11. 模型边界

Round1 当前模型可以直接预测：

```text
dim=1
input_stride=1
output_stride=1
block_dim=1..56
GM/UB contiguous
```

输入参数应为：

```python
branch = "le2" if block_dim <= 2 else "gt2"
params = model["dtype_models"][dtype]["branches"][branch]
cycles = params["alpha"] + bytes_per_core / params["T_bytes_per_cycle"]
```

不能直接用于：

- 非连续 GM
- 非连续 UB
- 二维及更高维
- `block_dim>56`
- broadcast
- store 路径
- 未验证的地址偏移和对齐组合

后续扩展时，应将 Round1 连续结果作为基础项，而不是把 stride 或 shape
影响重新吸收到 `alpha` 和 `T` 中。
