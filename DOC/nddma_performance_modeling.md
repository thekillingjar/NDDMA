## 整体建模流程
1. 首先对一维度多核连续NDDMA搬运进行建模(round1)
2. 然后对于一维建模进行扩展，实现支持非连续建模公式(round2)
3. 推测多维情况下是采用了连续建模+非连续乘法项的形式，对其进行建模(round3)
4. 对于常用的特殊case（二维转置）进行场景特化建模(round4)

### 1 一维度多核连续模型

#### 1.1 建模思路

首先，我们在不同block_dim对情况下观察数据量(B在一维情况下等于output_dim)和cycles(N_{base})的关系，如图所示，在固定的block_dim的情况下，$B$与$N_{base}$呈现线性关系。

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

#### 4.3 整体效果