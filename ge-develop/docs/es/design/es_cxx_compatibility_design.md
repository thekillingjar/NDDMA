# 历史原型库设计文档（ES 场景）

## 0. 阅读指引

- 本文聚焦 ES 场景：`gen_esb` 如何消费历史原型数据并生成 ES API（重点是第 4 章重载生成与二义性处理）。
- 历史原型库的**协议性内容**（定位/目录/格式/生成与发布/演进规则），见[协议文档](history_op_registry_protocol.md)。

## 0.1 术语（最小集合）

- 历史原型库：跨版本归档算子 IR 原型的结构化数据与协议。
- 结构化数据：`index.json` / `metadata.json` / `operators.json` 三类文件及其目录组织。
- 生产者：生成结构化数据的工具（当前固定为 `gen_esb --extract-history`）。
- 消费者：读取结构化数据的任意工具（当前主要是 `gen_esb`，未来可扩展到其他工具链）。
- 一年窗口/兼容周期：消费者依据版本元信息选择纳入对比/生成的版本范围。
- Ops run 包：Ops 安装后的运行包形态（下文若出现“run 包”均指 Ops run 包）。

## 1. 概述

### 1.1 背景

在 ES (Eager Style) 构图中，为了保证 C++ ES API 的向后 API 兼容性，生成工具需要掌握不同商发版本的 IR 原型演进情况。历史原型库负责沉淀并提供这些历史原型数据，供 `gen_esb` 在生成阶段对比差异、生成重载接口。

需要说明：
- `gen_esb` 是 ES 场景的工具：当前既是历史原型数据的主要消费者，也是结构化数据的生成工具（`--extract-history`）。短期内不提供新的生成工具。
- 历史原型库协议层不绑定 ES：未来可被其他消费者复用。本文只描述 ES 场景的消费与生成策略。


## 2. 模块架构与职责划分

### 2.1 模块划分

历史原型库系统由两个核心模块组成：

1. **历史原型库模块（Ops组件， Ops包）**：负责存储和管理历史原型数据
2. **消费、生成工具模块（gen_esb， GE仓， GE组件， GE包）**：
 - **消费者角色**：负责读取历史数据，生成 C++ 重载接口
 - **生产者角色**：负责生成历史原型库数据（复用 `gen_esb --extract-history`）

### 2.2 历史原型库模块职责

一句话：历史原型库模块的职责是**存储并以文件系统形式提供历史原型结构化数据**。

边界：
- 只提供数据与元信息（版本索引、算子原型等），不提供业务逻辑输出
- 不做二义性检测/兼容性判定/重载生成/代码生成

协议细节（目录/文件/字段）见[协议文档](history_op_registry_protocol.md)。

### 2.3 gen_esb 工具职责

**gen_esb 工具负责所有与代码生成相关的逻辑**。

#### 2.3.1 消费者角色

#### 2.3.1.1 数据读取

- 从历史原型库读取兼容周期内的历史版本 IR 定义，并解析 JSON 格式的历史原型数据

#### 2.3.1.2 版本对比分析

- 对比当前版本（当前版本直接从原型 .so 得来）与历史版本的 IR 定义差异，识别新增的可选输入、属性等
- **默认算子原型的变化都是兼容的**，不进行兼容性检查（GE的流程中已经有标准的兼容性检查流程，并且算子原型变化兼容本身就是一个基本要求，因此不做额外检查）

#### 2.3.1.3 重载生成

- 根据版本差异生成合法的重载签名同时添加必要的防呆接口，详见[章节](#4-重载生成与二义性处理)

#### 2.3.1.4 代码输出

- 生成 C++ 接口头文件（包含多个重载版本）
- 生成 C 接口（不重载，生成当前版本对应的原型）
- 生成 Python 接口（不重载，生成当前版本对应的原型）

#### 2.3.2 生产者角色

#### 2.3.2.1 历史原型库数据生成

- **复用 gen_esb 工具**：在商发版本发布前，使用 `gen_esb` 工具生成当前版本的原型数据
- 从当前版本的原型 .so 文件中提取 IR 定义
- 生成 JSON 格式的历史原型数据
- 输出到历史原型库目录结构中


### 2.4 模块间接口

#### 2.4.1 历史原型库提供的接口

历史原型库模块通过**文件系统接口**提供数据：

##### 2.4.1.1 分包的文件目录结构
协议与目录/数据格式详见 `history_op_registry_protocol.md`。本文仅使用其文件系统接口与关键文件名：
- `index.json`：版本列表（含发布日期等）
- `registry/<ver>/operators.json`：该版本的算子原型定义
- `registry/<ver>/metadata.json`：版本元信息（例如发布日期、分支名）

#### 2.4.2 gen_esb 工具的使用方式

**消费历史原型，用于生成 ES API 代码**：
```bash
gen_esb \
  --history-registry /path/to/registry/math \        # 历史原型库分包目录
  --output-dir /path/to/output \                # 输出目录
  --release-version 8.0.RC2                     # 当前版本号,用于查询兼容周期
```

历史原型库结构化数据的生成与发布属于协议侧内容，见[协议文档](history_op_registry_protocol.md) 的“生成与发布”与“附录 JSON 示例”。

#### 2.4.3 generate_es_package 函数的使用方式


[generate_es_package](https://gitcode.com/cann/ge/blob/master/docs/es/tools/generate_es_package_cmake_readme.md)是对gen_esb封装后的cmake函数，当前参数如下:

| 参数 | 必需性 | 说明 | 示例 |
|------|--------|------|------|
| `ES_LINKABLE_AND_ALL_TARGET` | 必需 | 对外暴露的库 target 名称 | `es_math`, `es_nn`, `es_cv` |
| `OPP_PROTO_TARGET` | 必需 | 算子原型库的 CMake target 名称 | `opgraph_math`, `opgraph_nn` |
| `OUTPUT_PATH` | 必需 | 产物输出的根目录 | `${CMAKE_BINARY_DIR}/output` |
| `EXCLUDE_OPS` | 可选  | 需要排除生成的算子 | `Add,Conv2D`|


**简化思路**

工程组通过定义一些全局的cmake变量, generate_es_package函数内部直接获取然后进行处理，无需generate_es_package新增一些参数； 

这样的好处是：
ops组件少一些感知，结合当前ops的分仓分包的背景，在generate_es_package函数内部处理可以避免各个分仓都要适配； 


**备选方案**

1. 新增如下的参数：

| 参数 | 必需性 | 说明 | 示例 |
|------|--------|------|------|
| `HISTORY_REGISTRY`| 可选  | 构建环境上的历史原型结构化数据目录 | `/path/to/registry/math` |
| `EXTRACT_HISTORY` | 可选  | 生成历史原型结构化数据， 默认不生成 | `ON`|
| `RELEASE_VERSION` | 可选  | 当前版本号 | `8.0.RC2`|

2. 调用方（Ops）传递对应的参数

**生成 ES API 代码**：
额外传递HISTORY_REGISTRY（可选，如果有历史原型的话）、RELEASE_VERSION（可选， 如果有历史原型的话）

**生成ES API && 历史原型库数据**：
额外传递HISTORY_REGISTRY（可选，如果有历史原型的话）、RELEASE_VERSION（必选， 当前历史原型数据对应的版本号）、EXTRACT_HISTORY（必选， 告知当前使能了历史原型结构化数据生成模式）

**推荐使用简化思路**


#### 2.4.4 其他需要的信息

以下信息如果需要显示传递的话，需要在gen_esb和generate_es_package内部处理对应的参数

release_date：商发时刻，branch_name：分支名 

### 2.5 职责边界总结

| 功能 | 历史原型库模块 | gen_esb 工具 |
|-----|--------------|-------------|
| 存储历史原型数据 | 负责 | 不负责 |
| 提供数据查询接口 | 负责（文件系统形式） | 不负责 |
| 生成历史原型库数据 | 不负责 | 负责 |
| 读取历史数据 | 不负责 | 负责 |
| 版本对比分析 | 不负责 | 负责 |
| 兼容性检测 | 不负责 | 不负责（默认兼容） |
| 二义性检测 | 不负责 | 负责 |
| 重载生成 | 不负责 | 负责 |
| 代码生成 | 不负责 | 负责 |

**核心原则**：
- **历史原型库 = 数据存储 + 数据提供**（纯数据层，作为 ops run 包的一部分）
- **gen_esb = 数据生成 + 数据读取 + 业务逻辑 + 代码生成**（业务逻辑层）

## 3. 数据流图

构建 Ops 包的“历史原型数据生成/合并/打包发布”流程属于协议侧内容，见[协议文档](history_op_registry_protocol.md) 的“生成与发布”。

**各组件配合流程图（ES 场景）**

工程组，算子，GE的配合如下：

![配合流程图](./figures/process_view.svg)


具体来说：

**商发版本初次构建时** 

工程组传递信息（编译宏的方式），告诉ge完成 1，2 两部分逻辑：

1. 根据历史原型的结构化数据和当前原型 .so 中的原型，生成算子粒度的 C++ 重载接口

2. 历史原型库结构化数据生成(为了简化第三步ops的处理逻辑， 此步骤生成当前版本历史原型库结构化数据之后会把已有的历史原型库结构化数据合并一起输出到指定目录)

3. ops组件对1和2的产物统一打包，完成ops包的构建


**非商发的构建或者商发版本日常构建时**

完成上述商发流程中的1，3两部分逻辑



## 4. 重载生成与二义性处理

### 4.1 问题根源

1. C++ 要求默认参数必须在参数列表末尾，基本类型的可选参数会存在重载二义性
2. `EsTensorLike` 支持从标量类型（`int64_t`、`float`）隐式转换，使得标量值可以同时匹配 `EsTensorLike` 参数和 `int64_t` 参数，产生二义性。


### 4.2 解决方案

#### 4.2.1 设计约束（用户侧语义约定）

本章节目标：在不引入“版本命名空间”的前提下，让历史重载接口可编译、无二义性，并尽量保持用户调用习惯稳定。

核心事实（来自 `EsTensorLike` 定义）：
- `EsTensorLike(const std::nullptr_t)`：可用 `nullptr` 表达“该输入未连边”（可选输入语义）
- `EsTensorLike(const int64_t/float/vector<...>)`：标量/向量会被视为常量输入（隐式构造）

核心约束（用户侧语义约定，面向“保留 v1、生成多版本重载”的策略）：
- 当新增可选输入 `xo2` 时：
  - **不使用 xo2（未连边）**：固定使用 v1 形态（例如 `Foo(x, xo1[, attr...])`），保证最前向兼容
  - **使用 xo2（连边）**：使用 v2 形态（例如 `Foo(x, xo1, xo2[, attr...])`）
- 为避免 `Foo(x, xo1)` 在 v1/v2 同时存在时产生二义性：v2 的 `xo2` 必须是**必选参数**（不得提供 `=nullptr` 默认值）。
- 同时生成防呆：对“v2 的 `xo2` 位置传 `nullptr`”给出 deprecated+delete 的明确诊断，提示用户“未连边请用 v1”。
- 当算子同时存在“可被标量字面量占位的属性参数”（如 `int64_t a=0`）时：
  - 此时 `Foo(x, xo1, 0)` 固定解释为属性（如 `a=0`），不允许把 `0` 当作新增输入 `xo2`
  - 若要把标量/向量作为 `xo2` 传入，必须先构造成 `Const/Scalar/Vector`（如 `EsGraphBuilder::CreateScalar/CreateVector/CreateConst`）再传入

#### 4.2.2 方案A：同命名空间、多版本重载（推荐）

方案A的核心目标：在同一命名空间内，通过一组可编译且无二义性的重载集合，兼顾后向兼容与前向兼容。

##### 4.2.2.1 A1：新增输入仍用 `EsTensorLike`，但 v2 中 `xo2` 必选 + `nullptr` 防呆

适用场景：新增可选输入 `xo2` 不会与现有参数产生“标量字面量歧义”（例如属性是 `const char*` / `std::string` 等，不会与 `EsTensorLike(int64_t)` 冲突）。

做法：
- 保留 v1：用于表达 `xo2` 未连边（最前向兼容的调用形态）
- v2：新增 `xo2`，但把 `xo2` 生成为必选参数（无默认值），用于表达 `xo2` 已连边
- 防呆：额外生成 `std::nullptr_t` 的 deprecated+delete 重载，提示“xo2 未连边请用 v1”

示例（`const char*` 默认属性场景）：

```cpp
// v1：xo2 未连边
FORCE_INLINE EsTensorLike Foo(const EsTensorLike &x, const EsTensorLike &xo1,
                            const char *mode = "xx");

// v2：xo2 已连边（xo2 必选，避免 Foo(x, xo1) 二义性）
FORCE_INLINE EsTensorLike Foo(const EsTensorLike &x, const EsTensorLike &xo1,
                            const EsTensorLike &xo2, const char *mode = "xx");

// 防呆：若 xo2 未连边，请直接调用 v1
[[deprecated("xo2 未连边请调用 v1: Foo(x, xo1[, mode])；若 xo2 需连边请传入有效 Tensor/Const。")]]
FORCE_INLINE EsTensorLike Foo(const EsTensorLike &x, const EsTensorLike &xo1,
                            std::nullptr_t /*xo2*/, const char *mode = "xx") = delete;
```

##### 4.2.2.2 A2：标量字面量歧义时切换 `TensorHolder`（兜底消歧）

触发条件（典型）：新增可选输入 `xo2` 与“标量属性默认参数”共存，会导致 `Foo(x, xo1, 0)` 这类调用出现歧义风险（`0` 既可能被当作 `xo2` 的标量常量，也可能被当作属性）。

做法：将新增输入的参数类型改为 `TensorHolder` 且无默认值；用户若需要把标量/向量作为 `xo2`，必须先 `CreateScalar/CreateVector/CreateConst`。

```cpp
// v1 版本（保留，用于后向兼容）
FORCE_INLINE EsTensorLike Foo(const EsTensorLike &x, const EsTensorLike &xo1, int64_t a=0);

// v2 版本（兜底：新增输入 xo2 变为必选 + TensorHolder 消歧）
FORCE_INLINE EsTensorLike Foo(const EsTensorLike &x, const EsTensorLike &xo1,
                            const TensorHolder &xo2, int64_t a=0, int64_t b=0);
```

#### 4.2.3 方案B：版本命名空间（备选）

方案B把不同版本的 API 放到不同命名空间（例如 `ge::es` 与 `ge::es::v2`），从而避免同名重载集合膨胀与二义性：

```cpp
// v1 版本
namespace ge::es {
FORCE_INLINE EsTensorLike Foo(const EsTensorLike &x, const EsTensorLike &xo1, const char *mode = "xx");
}

// v2 版本（新命名空间）
namespace ge::es::v2 {
FORCE_INLINE EsTensorLike Foo(const EsTensorLike &x, const EsTensorLike &xo1,
                            const EsTensorLike &xo2 = nullptr, const char *mode = "xx");
}
```

#### 4.2.4 方案对比与取舍

| 维度 | 方案A（同命名空间，多版本重载） | 方案B（版本命名空间） |
|---|---|---|
| 重载二义性风险 | 低：A1 默认 + A2 必要时 TensorHolder 兜底消歧 | 低：命名空间隔离 |
| 用户调用习惯稳定性 | 高：`Foo(x, xo1[, attr...])` 始终代表“未连边” | 中：调用方需显式选择命名空间 |
| 前向兼容引导 | 可用 deprecated+delete 强制用户“未连边走 v1” | 主要依赖文档规范 |
| Runnable Dump（C++） | 低成本：dump 只需判断“是否连边”选择 v1/v2 形态 | 高成本：dump 需要确定命名空间/版本，并处理依赖 |
| 动态库 ABI | 仍无法保证（命名空间不影响底层 C ABI） | 仍无法保证 |

综合取舍：本设计选择方案A。

#### 4.2.5 重载生成策略（方案A：A1/A2）

本小节给出可落地的生成规则（用于 `gen_esb`），目标是：在“一年窗口”的版本链上，生成一组**最小且无二义性**的 C++ 接口集合。

版本链说明：
- 版本链按版本元信息中的发布时间（例如 `metadata.json.release_date`）排序得到 `V0 < ... < Vn`；一年窗口的筛选策略由 `gen_esb` 根据当前版本号与发布日期计算得到。

术语定义：
- 抽取存在性：对某个版本的算子原型，判断某个输入/属性在 IR 原型中是否“存在”（以 IR 原型是否声明该 name 为准；可选/动态输入同样视为存在）。
- 新增点：在相邻两版本对比时，满足“新版本存在、旧版本不存在”的输入/属性。
- 首次出现：在版本序列（从旧到新）中，某个输入/属性第一次从“不存在”变为“存在”的版本。

备注（新增点 vs 首次出现）：
- 若 IR 原型在一年窗口内只发生“单调兼容变化”（仅新增可选输入/带默认属性），则相邻版本的“新增点”等价于版本链上的“首次出现”。
- 但我们还需覆盖未来可能出现的兼容变化（例如必选→可选），这类变化不一定体现为 exists 从 0→1；因此实现上仍保留
  “新增点（exists 变化）”与“首次触发生成（可能来自 exists/required/default 的变化）”的区分。

生成方法（每个算子独立执行；用于指导 `gen_esb` 实现）：
- 输入：一年窗口版本链 `V0 < V1 < ... < Vn` 的 IR 原型（inputs/attrs 及默认值）。
- 输出：一组**最小、无二义性、带防呆**的 C++ 接口签名集合（含 Runnable Dump 生成约束）。

阶段0：抽取存在性矩阵（见第 8 章附录《存在性矩阵与二义性检测示例》）
- 对每个版本 `Vi` 抽取 `exists_input[Vi][name]`、`exists_attr[Vi][name]`。
- 对相邻版本对 `(Vi-1, Vi)` 计算新增点 `new_inputs(Vi)`、`new_attrs(Vi)`。

阶段1：候选签名生成（先生成、后判定；阶段2 失败则回到这里收敛；不在这里穷举 case）
- 仅新增可选属性（有默认值）：不新增重载；把默认属性追加到**最新接口**参数末尾即可。
- 新增可选输入：先判定是否**可安全合并为单签名（A0）**，否则进入“保留 v1 的收敛链（尝试0→A1→A2）”：
  - 安全合并（A0）判定（同时满足时可合并，且不再保留 v1）：
    - 新增输入追加到末尾，且用 `= nullptr` 能准确表达“未连边”语义
    - 不存在“标量字面量可能被误当作新增输入”的风险（典型触发：同时存在 `int64_t a=0` 这类标量默认属性位）
    - 不需要用“保留 v1 + 防呆”来约束 dump/用户写法（例如不需要禁止 `Foo(..., nullptr)` 占位）
    - 通过后：生成单签名 `Foo(..., xo2=nullptr, ...)`，不再生成 v1；该场景不涉及重载二义性判定（阶段2可跳过）。
  - 不满足安全合并：进入收敛链（保留 v1）：
    - 尝试0（语义优先）：生成 v1，同时让 v2 的新增输入 `xo2` 保持可选（`xo2 = nullptr`）；若阶段2判定 v1/v2 存在潜在二义性，则放弃并升级到 A1。
    - A1（收敛1）：保留 v1 表达“未连边”；v2 新增输入 `xo2` 采用 `EsTensorLike` 且必选（无默认值），表达“已连边”；并生成 `nullptr` 防呆（见 4.3）。
    - A2（收敛2/兜底）：若 A1 仍可能与标量字面量发生重载歧义，则把 v2 的 `xo2` 切换为 `TensorHolder` 必选，并要求 dump/用户显式 `CreateScalar/CreateVector/CreateConst`。

阶段2：二义性判定（生成侧必须执行；失败则回到阶段1收敛）
- Gate1（候选对筛选）：对任意两条候选接口 `f/g`，计算可调用实参个数区间 `Rf=[req_f, tot_f]`、`Rg=[req_g, tot_g]`（`req`=必选参数数量）。若 `Rf ∩ Rg` 为空，则不可能二义性；否则进入 Gate2。
- Gate2（典型实参集合检查，保守判定）：在 `k ∈ (Rf ∩ Rg)` 的调用里，检查是否存在某个**典型实参**能在同一位置同时匹配 `f/g` 的形参类型。建议至少覆盖两类：
  - 可隐式转换为 `EsTensorLike`：`nullptr`、`0`、`0.0`（对应 `EsTensorLike(std::nullptr_t/int64_t/float)`）。
  - 不可隐式转换为 `EsTensorLike`：`"xx"`（`const char*`，用于验证“属性字符串”不会与输入位竞争）。
  若存在可同时匹配的典型实参，则判定为潜在二义性：禁止该组合；回到阶段1升级策略（尝试0→A1→A2），或调整签名集合。说明：该检查是工程上的保守近似，不追求完全等价于 C++ 的重载决议规则。

阶段3：防呆与导出约束落地
- 若最终集合包含“v2 新增输入必选”的重载（A1/A2），则自动生成 `std::nullptr_t` 的 deprecated+delete 防呆重载（见 4.3），并要求 Runnable Dump 不输出 `(..., nullptr, ...)` 占位形态。
- A0（单签名且新增输入默认 `= nullptr`）不生成上述 `nullptr` 防呆；Runnable Dump 可省略该输入以利用默认值。

最小示例（两种场景对比）：

A) 仅新增可选输入，且无标量字面量歧义（默认方案：v2 令 xo2 必选 + 防呆）
- v1：`Foo(x, xo1[, mode])`（xo2 未连边）
- v2：新增可选输入 `xo2`
生成：保留 v1 + 生成 v2（xo2 必选）+ 生成 nullptr 防呆
- `Foo(x, xo1[, mode])`
- `Foo(x, xo1, xo2[, mode])`
- `Foo(x, xo1, nullptr[, mode]) = delete`（deprecated 提示用 v1）

B) 新增可选输入 + 存在标量属性默认参数（默认方案会与字面量冲突，触发兜底方案A）
- v1：`Foo(x, xo1, a=0)`
- v2：新增可选输入 `xo2`
生成：保留 v1 + 新增 v2 重载（`TensorHolder` 消歧）
- `Foo(x, xo1, a=0)`
- `Foo(x, xo1, TensorHolder xo2, a=0, b=0)`

### 4.3 防呆机制（方案A）

防呆的目标：主要用于阻止“破坏向前兼容性”的占位写法，并给出明确迁移提示。

说明：只要新增输入在 v2 中是必选参数（无论类型是 `EsTensorLike` 还是 `TensorHolder`），就建议生成 `std::nullptr_t` 的 deprecated+delete 防呆重载，用于拦截用户误把 `nullptr` 当成“未连边”的写法，并提示回落到 v1。
补充：A0 场景（合并为单签名且新增输入默认 `= nullptr`）不需要生成 `nullptr` 防呆重载。

#### 4.3.1 `nullptr` 占位防呆（对“v2 新增输入必选”的重载生效）

触发条件：当某条 v2 重载因为“新增可选输入”而引入了新的输入形参 `xo2`，且该参数在 v2 中为必选参数（无默认值），则额外生成一条同名防呆重载。

签名形态：将该新增输入位置替换为 `std::nullptr_t`，并 `= delete`；必要时叠加 `[[deprecated("...")]]` 给出更友好的诊断。

示例：

```cpp
// 正常重载
FORCE_INLINE EsTensorLike Foo(const EsTensorLike &x, const EsTensorLike &xo1,
                            const TensorHolder &xo2, int64_t a=0, int64_t b=0);

// 防呆重载：禁止 nullptr 作为占位（避免误以为 nullptr 表示“未连边”）
[[deprecated("此算子新增输入 xo2 采用 TensorHolder 兜底消歧：若不使用 xo2 请调用旧重载 Foo(x, xo1[, a])；若 xo2 为标量/向量常量请使用 EsGraphBuilder::CreateScalar/CreateVector/CreateConst 显式构造。")]]
FORCE_INLINE EsTensorLike Foo(const EsTensorLike &x, const EsTensorLike &xo1,
                            std::nullptr_t /*xo2*/, int64_t a=0, int64_t b=0) = delete;
```

#### 4.3.2 Runnable Dump 规则（与两种方案对应）

- A0（单签名，新增输入默认 `= nullptr`）：
  - xo2 未连边：dump 可直接省略 xo2（利用默认值）
  - xo2 已连边：dump 直接传入 xo2
- 默认方案（`EsTensorLike` 作为新增输入类型，v2 中 xo2 必选）：
  - xo2 未连边：dump 输出 v1 形态 `Foo(x, xo1[, attr...])`（不输出 `Foo(x, xo1, nullptr, ...)`）
  - xo2 已连边：dump 输出 v2 形态 `Foo(x, xo1, xo2[, attr...])`
- 兜底方案A（`TensorHolder xo2`）：
  - xo2 若为标量/向量常量：dump 必须先输出 `builder.CreateScalar/CreateVector/CreateConst`，再把产物作为 `xo2` 传入
  - 禁止 dump 用 `Foo(x, xo1, 0)` 这类标量字面量形态表达 `xo2`（该形态固定用于属性表达，如 `a=0`）

## 5. ES 集成要点（最小）

- 历史原型库的目录/格式/归档流程属于协议内容，见[协议文档](history_op_registry_protocol.md)。
- ES 场景下的主要集成点是 `generate_es_package` 对 `gen_esb` 的封装（参数传递与构建流程），以及第 4 章定义的“重载生成与二义性处理规则”。

## 6. 特殊情况处理

### 6.1 首次启用历史原型库

**场景**：在某个版本（如 8.0.RC1）首次启用历史原型库，之前的版本未归档

**处理策略**：
1. 将当前版本作为"基线版本"
2. 仅生成当前版本的 API，不生成重载
3. 在后续版本中，以基线版本为起点生成重载

具体来说，2026/3/30版本规划为首次启用历史原型库的版本，但是2025/12/30是es构图的首次启动版本，因此330商发之前，需要先生成2025/12/30历史原型数据，然后生成重载版本的c++接口，同时归档330的历史原型数据

### 6.2 算子在历史版本中不存在

**场景**：当前版本新增的算子，在历史版本中不存在

**处理策略**：
- 仅生成当前版本的 API
- 不生成重载（因为没有历史版本）
- C++ 接口与 C 接口一一对应

## 7. 总结

历史原型库是 ES 构图系统中支持 C++ API 兼容性的关键基础设施：

1. **核心作用**：存储历史版本的 IR 定义，供 `gen_esb` 生成 C++ 重载接口
2. **关键原则**：
   - C 接口：单一版本，包含所有参数
   - C++ 接口：多个重载版本，往前一年内的所有版本都要生成
   - Python 接口：单一版本，使用关键字参数
   - 所有接口最终都调用同一个 C 实现
3. **数据格式**：采用 JSON 格式，作为 ops run 包的一部分打包发布
4. **重载生成**：基于历史版本差异，为可选输入的演进生成重载
5. **二义性处理**：检测并规避重载二义性，保证生成的代码可编译
6. **维护流程**：商发前使用 `gen_esb` 工具生成数据，打包到 ops run 包中
7. **兼容性**：默认算子原型的变化都是兼容的，不进行兼容性检查；兼容性保证由 IR 语义运行时兼容处理机制负责

通过历史原型库，ES 构图系统能够在保证易用性的同时，提供可靠的 API/ABI 兼容性保障。

## 8. 附录：存在性矩阵与二义性检测示例

### A.1 存在性矩阵（输入/属性）

一年窗口内版本为 `V1 < V2 < V3`，该算子在各版本出现过的输入名并集为 `{x, xo1, xo2}`，属性名并集为 `{a, b}`。

输入存在性矩阵（`1` 表示该版本 IR 中存在该输入定义）：

| version | x | xo1 | xo2 |
|---|---:|---:|---:|
| V1 | 1 | 1 | 0 |
| V2 | 1 | 1 | 1 |
| V3 | 1 | 1 | 1 |

属性存在性矩阵：

| version | a | b |
|---|---:|---:|
| V1 | 1 | 0 |
| V2 | 1 | 0 |
| V3 | 1 | 1 |

由此可得：
- 新增点：`new_inputs(V2)={xo2}`，`new_attrs(V3)={b}`
- 首次出现：`xo2` 首次出现在 `V2`，`b` 首次出现在 `V3`

### A.2 Gate1/Gate2 的“区间交集”与二义性示例

两条候选重载：
- f1: `Foo(x, xo1, int64_t a=0)`，可调用范围 `R1=[2,3]`
- f2: `Foo(x, xo1, TensorHolder xo2, int64_t a=0)`，可调用范围 `R2=[3,4]`

Gate1（交集）：`R1 ∩ R2 = [2,3] ∩ [3,4] = [3,3]`，说明只有当调用传入 3 个实参时，两者才可能同时成为候选。

Gate2（实参可同时匹配）：
- 调用 `Foo(x, xo1, 0)`：
  - 对 f1：`0` 可转 `int64_t`，可匹配
  - 对 f2：`0` 不能转 `TensorHolder`，不可匹配
  - 因此最终不二义性，且固定解释为 `a=0`

若把 f2 的 `xo2` 类型改为 `EsTensorLike`（且存在 `EsTensorLike(int64_t)` 隐式构造），则 `Foo(x, xo1, 0)` 会同时可匹配 f1 与 f2，从而发生二义性；这就是“冲突时方案A（TensorHolder 兜底）”的触发原因。


### A.3 新增可选输入（EsTensorLike 必选 + nullptr 防呆）

当 v1/v2 需要同时存在时，v2 不能把新增输入写成 `xo2=nullptr`（否则 `Foo(x, xo1)` 会二义性）。推荐形态是：
- v1：表达 `xo2` 未连边
- v2：`xo2` 必选，表达 `xo2` 已连边
- 防呆：`std::nullptr_t` 重载提示用户回落到 v1

示例：
- v1: `Foo(x, xo1, const char* mode="xx")`
- v2: `Foo(x, xo1, EsTensorLike xo2, const char* mode="xx")`
- 防呆: [deprecated("描述信息")] `Foo(x, xo1, nullptr, const char* mode="xx") = delete`
