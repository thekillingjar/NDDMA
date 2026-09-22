# NDDMA2 Round1 一维单核与多核连续建模

## 1. 目标

NDDMA2 Round1 建立多核一维连续 NDDMA 的基础模型

## 2. 目录和运行入口

Round1 脚本：

```text
NDDMA2/Modeling/Data_collection/round1/scripts/
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

不是 `logical_total_bytes`。模型同时区分每核固定开销 `h` 和不随核数
变化的全局固定开销 `H`。

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

每个分支内联合拟合以下单核/每 block 周期模型：

```text
cycles(dtype, block_dim, B)
    = B * (block_dim / T(dtype) + h(dtype)) + H(dtype)
```

其中：

```text
B = bytes_per_core
```

等价地：

```text
cycles = cycles_per_byte * block_dim * B + h * B + H
cycles_per_byte = 1 / T
```

每个 `(dtype, block_dim)` 有至少两个 bytes 档位，因此可以拟合截距和斜率。

### 5.2 分段模型

Round1 最终把 `block_dim` 分为两个分支：

```text
block_dim <= 2:
    cycles = bytes_per_core * (block_dim / T_1(dtype) + h_1(dtype)) + H_1(dtype)

block_dim > 2:
    cycles = bytes_per_core * (block_dim / T_2(dtype) + h_2(dtype)) + H_2(dtype)
```

因此在 `bytes_per_core` 固定且较大时，`block_dim > 2` 的单核
`cycles` 对核数呈线性关系。该线性项的斜率为
`bytes_per_core / T_2`，`h_2` 表示与 `bytes_per_core` 相关但不随核数
变化的斜率偏置，`H_2` 是不随字节数和核数变化的固定项。

对应关系：

```text
le2: block_dim = 1, 2
gt2: block_dim = 3..56
```

### 5.3 分支参数的取得方式

对于某个 dtype 和某个分支：

1. 固定每个 `block_dim`，对不同 `bytes_per_core` 拟合：
   `cycles = slope(block_dim) * bytes_per_core + H(block_dim)`。
2. 在每个分支内拟合 `slope(block_dim)` 与核数的线性关系：
   `slope(block_dim) = block_dim / T + h`。
3. `T = 1 / slope_slope`，`h = slope_intercept`。
4. 分支 `H` 取该分支内各 `H(block_dim)` 的算术平均。

### 5.4 参数单位

```text
H   : cycles
h   : cycles/byte
T   : bytes/cycle
```

预测时推荐使用：

```python
predicted = bytes_per_core * (block_dim / T + h) + H
```


## 6. 运行命令

从 `NDDMA2` 根目录执行。

### 6.1 完整流程

```bash
python3 Modeling/Data_collection/round1/scripts/e2e.py all \
  --msprof-bin "$(which msprof)"
```

执行顺序：

```text
collect -> fit -> draw
```

### 6.2 生成、编译和采集

```bash
python3 Modeling/Data_collection/round1/scripts/e2e.py collect \
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
python3 Modeling/Data_collection/round1/scripts/e2e.py fit
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
python3 Modeling/Data_collection/round1/scripts/e2e.py fit \
  --measurement-csv /path/to/measurements.csv
```

### 6.4 只绘图

```bash
python3 Modeling/Data_collection/round1/scripts/e2e.py draw
```

绘图读取：

```text
NDDMA2/Modeling/Ana/round1/round1_1d_single_core_predictions.csv
```

### 6.5 自定义路径

```bash
python3 Modeling/Data_collection/round1/scripts/e2e.py all \
  --msprof-bin "$(which msprof)" \
  --run-output-dir /path/to/collection \
  --model-output-dir /path/to/model
```

## 8. 拟合 JSON

输出文件：

```text
NDDMA2/Modeling/Ana/round1/round1_1d_single_multi_core_model.json
```

JSON 保留建模公式、最终参数和整体误差指标，不再输出每个核数的局部拟合过程。

```json
"model": "NDDMA_ROUND1_1D_PIECEWISE_SINGLE_MULTI_CORE"
```

### 8.1 顶层结构

```json
{
  "model": "...",
  "formula": {},
  "parameters": {},
  "metrics": {}
}
```

### 8.2 `formula`

示例：

```json
"formula": {
  "name": "arbitrary_dim_multicore_1d_contiguous_base",
  "split_block_dim": 2,
  "le2": "cycles = bytes_per_core * (block_dim / T_1(dtype) + h_1(dtype)) + H_1(dtype)",
  "gt2": "cycles = bytes_per_core * (block_dim / T_2(dtype) + h_2(dtype)) + H_2(dtype)",
  "metric": "cycles is single-core/per-block cycles",
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

### 8.3 `parameters`

每种 dtype 保存两段公式的最终参数：

```json
{
  "int32_t": {
    "T_1": 0.0,
    "h_1": 0.0,
    "H_1": 0.0,
    "T_2": 0.0,
    "h_2": 0.0,
    "H_2": 0.0
  }
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `T_1` | `block_dim<=2` 分支平均搬运速率，单位 bytes/cycle |
| `h_1` | `block_dim<=2` 分支字节斜率偏置，单位 cycles/byte |
| `H_1` | `block_dim<=2` 分支不随核数变化的固定项，单位 cycles |
| `T_2` | `block_dim>2` 分支平均搬运速率，单位 bytes/cycle |
| `h_2` | `block_dim>2` 分支字节斜率偏置，单位 cycles/byte |
| `H_2` | `block_dim>2` 分支不随核数变化的固定项，单位 cycles |

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
| `actual_cycles` | 聚合后的实际单核 cycles |
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
block_dim
```

纵轴使用单核周期：

```text
cycles
```

每种 dtype 只绘制两组 shape：

```text
min bytes_per_core
max bytes_per_core
```

这样可以直接观察固定单核数据量时，单核 cycles 随设置的
`block_dim` 的变化关系。

### 10.3 图中元素

- 实线/实心点：`actual_cycles`
- 虚线/空心点：`predicted_cycles`
- 蓝色：最小 `bytes_per_core`
- 红色：最大 `bytes_per_core`

注意：Round1 的 `cycles` 口径是单核/每 block cycles，不是总 cycles。

预测点和实测点按 `block_dim` 连线；每条线内部的 `bytes_per_core`
固定。

### 10.4 判断方法

理想情况：

- 虚线预测点接近同色实线实测点
- 单核/双核点与多核点都没有明显系统性偏差
- 最小和最大 shape 都没有随核数单向发散的残差

常见异常：

| 图形现象 | 可能含义 |
| --- | --- |
| `block_dim<=2` 点整体偏离 | `le2` 分支参数不适合单核/双核 |
| `block_dim>2` 点整体偏离 | `gt2` 分支参数不适合多核 |
| 大字节量误差变大 | `T` 不准确或线性假设不足 |
| 小字节量误差大、大字节量较好 | `H` 不准确 |
| 某些核数单独异常 | 该核数存在额外调度或硬件行为 |
| 两个分支都呈明显弯曲 | 需要进一步引入连续的 `T(block_dim)` 或 `H(block_dim)` 模型 |

图像只提供直观诊断，逐点误差应查看 predictions CSV 的：

```text
error_cycles
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
params = model["parameters"][dtype]
if branch == "le2":
    cycles = bytes_per_core * (block_dim / params["T_1"] + params["h_1"]) + params["H_1"]
else:
    cycles = bytes_per_core * (block_dim / params["T_2"] + params["h_2"]) + params["H_2"]
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
影响重新吸收到 `H` 和 `T` 中。
