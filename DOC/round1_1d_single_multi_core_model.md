# NDDMA2 Round1 一维多核连续建模

## 1. 目标

Round1 复用 `HW_GE_ATT` round2 的 A+C 数据集，建立一维连续 NDDMA 的
分段多核基础模型。

## 2. 运行入口

```text
NDDMA/Modeling/Data_collection/round1/scripts/
├── e2e.py
├── collect.py
├── fit.py
└── draw.py
```

默认产物位于：

```text
NDDMA/Modeling/Ana/round1/
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

覆盖 `int8_t`、`int16_t`、`int32_t`、`int64_t`，其中
`block_dim` 表示核数。

## 4. 数据集

Round1 使用与 `HW_GE_ATT/NDDMA/Modeling/Data_collection/round2/scripts/plan_common.py`
一致的 A+C fit 数据：

| 组别 | `bytes_per_core` 来源 | `block_dim` | 用途 |
| --- | --- | --- | --- |
| A | `STAGE1_X_BY_DTYPE` | `1,2,4,8,16,32,56` | 代表性核数拟合 |
| C | `C_X_BY_DTYPE` | `1..56` | 全核数拟合 |

当前配置数：

```text
A = 483
C = 1904
A + C = 2387
```

模型中的字节量为：

```text
bytes_per_core = 单个核负责的搬运量
logical_total_bytes = bytes_per_core * block_dim
```

Factor 文件名保留 `round1_1d_single_core_factor.csv` 以兼容现有工作流，
但内容已经是 round2 A+C 数据。

## 5. 建模公式

核数在 `block_dim=3` 处分段：

```text
block_dim <= 2:
    cycles = bytes_per_core / T_1(dtype) + H_1(dtype)

block_dim > 2:
    cycles = bytes_per_core / T_2(dtype) + H_2(dtype)
```

这里的 `cycles` 是单次 NDDMA 的 `mte2` 周期，`T` 单位为
`bytes/cycle`，`H` 单位为 `cycles`。

对每个 dtype、每个分支，将 A+C 的全部 fit 样本联合进行最小二乘拟合：

```text
cycles = bytes_per_core / T + H
```

不再使用旧模型中的 `h` 项，也不再先按每个 `block_dim` 拟合局部斜率。
