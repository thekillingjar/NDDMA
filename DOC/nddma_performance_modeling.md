## 整体建模流程
1. 首先对一维度多核连续NDDMA搬运进行建模(round1)
2. 然后对于一维建模进行扩展，实现支持非连续建模公式(round2)
3. 推测多维情况下是采用了连续建模+非连续乘法项的形式，对其进行建模(round3)
4. 对于常用的特殊case（二维转置）进行场景特化建模(round4)

### 1 一维度多核连续模型

#### 1.1 建模思路

首先，我们在不同block_dim对情况下观察数据量(B在一维情况下等于output_dim)和cycles(N_{base})的关系，如图所示，在固定的block_dim的情况下，$B$与$N_{base}$呈现线性关系。
  <table>
    <tr>
      <td><img src="figures/img1.svg" alt="图1" width="400"></td>
      <td><img src="figures/img2.svg" alt="图2" width="400"></td>
    </tr>
    <tr>
      <td><img src="figures/img3.svg" alt="图3" width="400"></td>
      <td><img src="figures/img4.svg" alt="图4" width="400"></td>
    </tr>
  </table>

  如果想每张图下面有标题：

  <table>
    <tr>
      <td align="center"><img src="figures/img1.svg" width="400"><br>图1</td>
      <td align="center"><img src="figures/img2.svg" width="400"><br>图2</td>
    </tr>
    <tr>
      <td align="center"><img src="figures/img3.svg" width="400"><br>图3</td>
      <td align="center"><img src="figures/img4.svg" width="400"><br>图4</td>
    </tr>
  </table>
并且如图2所示，我们在固定的B=61440的情况下，$block\_dim$和$N_{base}$在$block\_dim=2$处分段，并且两段都呈现出了近似线性的关系。

于是我们将其建模成1.2的形式

#### 1.2 建模形式
```math
N_{base} = \begin{cases}
B\cdot(block\_dim/T_1+h_1)+H_1, & block\_dim \le 2 \\
B\cdot(block\_dim/T_2+h_2)+H_2, & block\_dim > 2
\end{cases}
```

其中$B$为搬运的字节总数，$block\_dim$为核数, $h_1$为核冲突所造成的开销，$H_1$为固定总开销。其分成两段的主要原因，可能是当<=2时，数据在两个不同DIE上传输，造成的带宽抢占小。

#### 1.3 数据集

Round1 数据由 `Modeling/Data_collection/round1/scripts/collect.py` 生成，
默认 factor 文件为：

```text
Modeling/Ana/round1/round1_1d_single_core_factor.csv
```

固定配置：

```text
dim=1
input_stride=1
output_stride=1
enable_store=0
src_offset_elem=0
dst_offset_elem=0
kernel_repeat=200
execution_repeat_count=1
```

变量范围：

```text
dtype = int8_t, int16_t, int32_t, int64_t
block_dim = 1..56
```

每个样本的 `output_dims` 由 `bytes_per_core / dtype_size` 反推，
`logical_total_bytes = bytes_per_core * block_dim`。拟合自变量使用
`bytes_per_core`，不是所有核累加后的 `logical_total_bytes`。

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

#### 1.4 整体效果


### 2 一维度多核非连续模型

#### 2.1 建模思路

我们继承round1所建模的一维度多核连续模型$N_{base}$，我们首先分析单核情况下，GM连续和UB非连续分别对整体性能造成的影响。如图所示，GM非连续呈现出roofline的形式，并且在$input\_stride*dtype\_size=128$的位置发生分段，于是我们直接对$s=min(input\_stride, 128)$拟合出的$N_{G}$作为$N_{base}$的GM非连续惩罚项。

为了对$N_G$进行建模，我们采集数据绘制了在不同的$input\_stride$下D(output_dim)与cycles对关系，如图所示，发现在相同的$input\_stride$下，D与cycles呈现线性关系，于是我们分两段拟合$N_G(D, input\_stride)$，即首先固定input\_stride，拟合slope*D+d,然后再分别拟合slop和d与input\_stride的线性关系，最终建模形式如2.2所示。

我们接下来对UB非连续的建模，发现UB连续和非连续表现出不同的常数影响，于是我们可以直接将其与GM/UB同时非连续同一建模，因为UB=1时只有一个$N_G$惩罚项。如图所示，我们在$os=2$下，发现出现与上述$N_G$相似的情况，也采用相似的方式进行建模$N_{GU}$。

当然我们为了避免公式的复杂性，对建模公式进行了部分参数的删除以简化建模。

最后，我们引入多核惩罚项，如图所示，我们分析在UB非连续，不同input\_stride情况下，block\_dim和$\rho=(N_{act}-N_{base})/(N_G+N_{GU})$的关系，发现随着input\_stride增加增加，在block_dim=2处分段，与多核连续相似。因此我们构建$\rho$与input\_stride的线性模型。
#### 2.2 建模形式
```math
N_1'(B,is,os,k)=
\begin{cases}
N_G+N_{GU}, & k \le 2 \\
(N_G+N_{GU})\cdot \rho, & k > 2
\end{cases}
```

其中：

```math
s=\min(is\cdot dtype\_size,128)
```

```math
N_G=(a_1+a_2B)s
```

```math
N_{GU}=
\left((b_1+b_2s)+(b_3+b_4s)B\right)\cdot \min(1,os-1)
```

```math
\rho=
(c_1+c_2s)+\min(1,os-1)(c_3+c_4s)
```



#### 2.3 数据集

Round2 数据由 `Modeling/Data_collection/round2/scripts/collect.py` 生成，
默认 factor 文件为：

```text
Modeling/Ana/round2/round2_1d_noncontiguous_factor.csv
```

公共固定配置：

```text
dim=1
enable_store=0
kernel_repeat=200
execution_repeat_count=3
dtype = int8_t, int16_t, int32_t, int64_t
```

安全过滤：

```text
GM span <= 4 MiB
UB span <= 256 KiB
guard = 64 elems
```

枚举集合：

```text
OUTPUT_DIMS = 8, 16, 32, 64, 128, 512, 1024, 2048, 4196, 10001
BLOCK_DIMS = 1, 2, 4, 8, 15, 32, 56
INPUT_STRIDES = 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
OUTPUT_STRIDES = 2, 4, 6, 8, 10, 12, 14, 16, 32
BYTES_PER_CORE = 64, 128, 512, 1024, 4096, 16384
```

各组含义和过滤后配置数：

| group | 作用 | 核数/stride 组合 | 配置数 |
| --- | --- | --- | ---: |
| A | 单核 GM 非连续，拟合 `N_G` | `block_dim=1, output_stride=1, input_stride>=2` | 398 |
| B | 单核 UB 非连续，参与 `N_GU` | `block_dim=1, input_stride=1, output_stride>1` | 329 |
| C | 单核 GM/UB 联合非连续，拟合 `N_GU` | `block_dim=1, input_stride>=2, output_stride>1` | 3385 |
| F | 多核 base 验证 | `block_dim in {1,2,4,8,15,32,56}`, `(is,os) in {(1,1),(1,2),(1,16)}` | 476 |
| G | 多核 GM 非连续校准 | `output_stride=1`, 低档 `input_stride` | 840 |
| H | 多核 GM/UB 联合校准 | `output_stride=2`, `input_stride in {2,4,8,16,32,64,128,256}` | 1344 |
| I | 多核输出侧验证 | `(is,os) in {(2,4),(32,16),(128,32)}` | 448 |
| J | 指定单点验证 | `int32_t, block_dim=55, output_dim=9656, input_stride=3, output_stride=1` | 1 |

总 factor 配置数为 7221。实际参与当前 JSON metrics 的有效测量点为 6717，
原因是拟合脚本会按可用 profiling 结果聚合，并跳过缺失或无效测量。

按 dtype 的 factor 配置数：

| dtype | 配置数 |
| --- | ---: |
| `int8_t` | 1908 |
| `int16_t` | 1854 |
| `int32_t` | 1780 |
| `int64_t` | 1679 |

#### 2.4 整体效果



### 3 多维度模型

#### 3.1 建模思路

针对于多维度建模，我们推测其建模形式采用基本数据搬运+每维度惩罚项的形式。
#### 3.2 建模形式
##### 3.2.1 总数据量

每核每次搬运的总字节数为：

```math
B = \left(\prod_{j=0}^{D-1} ls_j\right)\cdot dtype\_size
```

其中 `dtype_size` 是单个元素的字节数。
##### 3.2.2 每一维的等效 stride

第 0 维直接使用原始最内侧 stride：

```math
\hat{is}_0 = is_0
```

```math
\hat{os}_0 = os_0
```

第 `j >= 1` 维使用相对连续偏移量建模，且 `+1` 在绝对值外侧：

```math
\hat{is}_j =
\left|
is_j - \sum_{t=0}^{j-1} ls_t\cdot is_t
\right| + 1
```

```math
\hat{os}_j =
\left|
os_j - \sum_{t=0}^{j-1} ls_t\cdot os_t
\right| + 1
```

这里的含义是：第 `j` 维的 stride 代价由它相对于所有更内侧轴连续展开后的偏移量决定。即使完全连续，等效 stride 也通过 `+1` 保持为合法的一维 stride 输入。

##### 3.2.3 多维统一公式

任意 `D` 维的多核预测 cycles 写为：

```math
N_D =
N_{base}(B,k)
+
\sum_{j=0}^{D-1} N_1'(B_j,\hat{is}_j,\hat{os}_j,k)
```

其中：

```math
B_j =
\frac{B}{\prod_{t=0}^{j-1} ls_t}
```

特别地：

```math
B_0 = B
```

也就是说，第 `j` 维修正项使用的是去掉更内侧 `0 ... j-1` 维之后的数据量。




#### 3.3 数据集

Round3 数据由 `Modeling/Data_collection/round3/scripts/collect.py` 生成，
默认 factor 文件为：

```text
Modeling/Ana/round3/round3_multidim_factor.csv
```

公共固定配置：

```text
dtype = int8_t, int16_t, int32_t, int64_t
block_dim=1
kernel_repeat=200
execution_repeat_count=3
enable_store=0
```

安全过滤与 Round2 一致：

```text
GM span <= 4 MiB
UB span <= 256 KiB
guard = 64 elems
```

数据来源和过滤后配置数：

| dim | 来源 | 形状枚举摘要 | layout 数 | 配置数 |
| --- | --- | --- | ---: | ---: |
| 2D | 旧 Round4/Round5 二维 transpose 类规划 | `d0 in {4,8,16,32,64,128,153,167,224}`, `d1 in {4,8,16,32,64,128,153,167}` | 3 | 858 |
| 3D | 旧 Round6 | `m in {2,4,8,16,32}`, `n in {4,8,16,32,64}`, `k in {4,8,16,32,64,128}` | 5 | 2825 |
| 4D | 旧 Round7 | `l3,l2 in {2,4,8,16}`, `l1 in {4,8,16,32}`, `l0 in {4,8,16,32,64,128}` | 3 | 3924 |
| 5D | 旧 Round7 | `l4,l3,l2 in {2,4,8,16}`, `l1 in {4,8,16,32}`, `l0 in {4,8,16,32,64,128}` | 3 | 10389 |

总配置数为 17996。

layout 维度：

| layout_pattern | 配置数 |
| --- | ---: |
| `continuous` | 5622 |
| `transpose_outer` | 286 |
| `transpose_outer_ub_gap` | 286 |
| `transpose_dim2` | 565 |
| `transpose_dim1_dim3` | 565 |
| `transpose_3d` | 565 |
| `gm_noncontiguous` | 565 |
| `transpose_dim1_dim4` | 1308 |
| `transpose_dim3_dim4` | 1308 |
| `transpose_dim1_dim5` | 3463 |
| `transpose_dim1_dim2` | 3463 |

按维度和 dtype 的配置数：

| dim | `int8_t` | `int16_t` | `int32_t` | `int64_t` |
| --- | ---: | ---: | ---: | ---: |
| 2D | 216 | 216 | 216 | 210 |
| 3D | 745 | 730 | 700 | 650 |
| 4D | 1107 | 1047 | 951 | 819 |
| 5D | 3477 | 2922 | 2304 | 1686 |

#### 3.4 整体效果

### 4 二维转置模型

#### 4.1 建模思路

该模型对应的是二维 UB 连续配置的专用模型：

```text
output_dims   = [M,N]
input_stride  = [is2,is1]
output_stride = [N,1]
```

该形式不替代第 3 节的任意维统一公式，而是作为 `[M,N]/[is2,is1]/[N,1]`
场景的二维特化估计。它和统一公式保持相同的 `N_base` 与一维修正项
`N_1'` 定义，只在外层二维结构上额外引入一个 `N_G2` 乘子。
这里 `os1=1`、`os2=N`，因此一维输出侧 gap 修正为 0。

判断该形式的使用条件：
1. 首先判断output_dims(ls)中是否只存在两个非1维度，然后提取对应的is和os，即[ls2, ls1]/[is2, is1]/[os2, os1],
2. 判断是否满足约束: is2 < is1, os2 = ls1, os1 = 1, 如果满足约束将使用该建模形式否则使用统一建模形式

```math
B=M N\cdot dtype\_size
```

```math
B_1=N\cdot dtype\_size
```

```math
N_{G1}^{uni}=N_1'(B_1,is_1,1,k)
```

其中 `N_{base}(B,k)` 与第 5.1 节一致，`N_1'(B_1,is_1,1,k)`
与第 5.2 节一致；也就是说 `k<=2` 和 `k>2` 的分段、
`rho` 多核修正都沿用任意维统一模型。Round5 在该二维模式下只新增
外层乘子 `N_G2`。

```math
N_{G2}=(g10+g11_M\cdot M)\cdot is_2+g00+g01_M\cdot M
```

```math
N_2=N_{base}(B,k)+N_{G1}^{uni}\cdot N_{G2}
```



`N_G2` 对应参数以 `NDDMA/Modeling/DOC/round5_ng2_model.json` 为准。
JSON 中的 `g11_per_m`、`g01_per_m` 与本文的 `g11_M`、`g01_M`
含义相同，都是对 `M` 的线性系数。此前本文表格沿用了
`round5_ng2_only_summary` 的旧摘要值，因此会与当前
`round5_ng2_model.json` 不一致；后续以本表和该 JSON 为准。



#### 4.2 数据集

Round4 数据由 `Modeling/Data_collection/round4/scripts/collect.py` 生成，
默认 factor 文件为：

```text
Modeling/Ana/round4/round4_2d_ub_contiguous_ng2_factor.csv
```

公共固定配置：

```text
dim=2
output_dims=[M,N]
input_stride=[is2,is1]
output_stride=[N,1]
is2 < is1
enable_store=0
kernel_repeat=200
execution_repeat_count=3
```

枚举集合：

```text
block_dim = 2, 4, 8, 32, 64
M = 2, 4, 8, 16, 32, 64, 128, 256
N = 8, 16, 32, 64, 128, 512, 2048
is2 = 1, 2, 4, 8, 16, 32
is1 = 2, 4, 8, 16, 32, 64, 128, 256, 1024
```

安全过滤：

```text
GM span([M,N],[is2,is1]) <= 4 MiB
UB span([M,N],[N,1]) <= 256 KiB
guard = 64 elems
```

总配置数为 39360。每个 `block_dim` 都有 7872 个配置：

| block_dim | 配置数 |
| ---: | ---: |
| 2 | 7872 |
| 4 | 7872 |
| 8 | 7872 |
| 32 | 7872 |
| 64 | 7872 |

按 dtype 的配置数：

| dtype | 配置数 |
| --- | ---: |
| `int8_t` | 10530 |
| `int16_t` | 10135 |
| `int32_t` | 9630 |
| `int64_t` | 9065 |

按 `is1*dtype_size` 是否超过 128 字节分为两类：

| split | 含义 | 配置数 |
| --- | --- | ---: |
| `calibration` | `is1*dtype_size <= 128`，用于主要参数校准 | 18850 |
| `high_stride_validation` | `is1*dtype_size > 128`，用于高 stride 外推验证 | 20510 |

#### 4.3 整体效果
