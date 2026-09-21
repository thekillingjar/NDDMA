# ES 模块所有权关系反转分析

## 问题描述

Python 层和 C++ 层的所有权关系是**反向**的：

### C++ 层所有权关系

```cpp
struct EsCGraphBuilder {
    std::list<std::unique_ptr<ResourceHolder>> resource_holder_;  // 拥有所有资源
    std::unique_ptr<ge::Graph> graph_;
    // ...
};

struct EsCTensorHolder {
    EsCGraphBuilder &owner_graph_builder_;  // 只是引用，不拥有
    ge::GNode producer_;
    int32_t producer_out_index_;
};
```

**关系**：`EsCGraphBuilder` **拥有**（owns） `EsCTensorHolder`

### Python 层所有权关系

```python
class GraphBuilder:
    _handle: EsCGraphBuilderPtr  # C 对象指针（不拥有）
    
class TensorHolder:
    _handle: EsCTensorHolderPtr  # C 对象指针（不拥有）
    _builder: GraphBuilder        # 强引用，拥有
```

**关系**：`TensorHolder` **拥有**（owns） `GraphBuilder`

## 为什么要这样设计？

### 原因：防止悬空指针

```python
def create_tensor():
    builder = GraphBuilder("my_graph")
    tensor = builder.create_const_float(1.0)
    return tensor  # builder 会被 GC

# 问题场景
t = create_tensor()
# 如果 TensorHolder 不持有 GraphBuilder：
# - builder 已被 GC，Python 对象销毁
# - builder.__del__() 调用 EsDestroyGraphBuilder()
# - 底层 C++ 的 EsCGraphBuilder 被析构
# - 底层 C++ 的 EsCTensorHolder 也被释放（因为被 GraphBuilder 拥有）
# - t._handle 现在是悬空指针!

# 当前设计（TensorHolder 持有 GraphBuilder）：
# - builder Python 对象被 GC，但因为 t._builder 持有引用，不会真正析构
# - 底层 C++ 对象保持有效
# - t._handle 仍然有效 
```

## 潜在问题分析

### 问题 1: 循环引用风险

#### 场景描述

```python
class GraphBuilder:
    def __init__(self):
        self._tensors = []  # 如果保存了 tensor 列表
    
    def create_const_float(self, value):
        tensor = TensorHolder._create_from(handle, self)
        self._tensors.append(tensor)  # ⚠️ 循环引用！
        return tensor

# 循环引用：
# GraphBuilder._tensors -> TensorHolder
# TensorHolder._builder -> GraphBuilder
```

#### 影响

- Python GC 无法自动回收（需要等待循环检测）
- 可能导致内存泄漏
- 对象析构延迟

#### 解决方案

```python
# 方案1: 不在 GraphBuilder 中保存 TensorHolder
class GraphBuilder:
    # ❌ 不要这样做
    # self._tensors = []
    pass

# 方案2: 使用弱引用
import weakref

class GraphBuilder:
    def __init__(self):
        self._tensors = []  # 保存弱引用
    
    def create_const_float(self, value):
        tensor = TensorHolder._create_from(handle, self)
        self._tensors.append(weakref.ref(tensor))  # 弱引用
        return tensor
```

**当前代码状态**：✅ 安全，GraphBuilder 没有保存 TensorHolder 列表

---

### 问题 2: 语义不一致

#### C++ 层的预期

```cpp
{
    EsCGraphBuilder builder("my_graph");
    auto tensor1 = builder.CreateConstFloat(1.0);
    auto tensor2 = builder.CreateConstFloat(2.0);
    // tensor1, tensor2 是原始指针，生命周期由 builder 管理
} // builder 析构，所有 tensor 也被释放
```

#### Python 层的实际行为

```python
def test():
    builder = GraphBuilder("my_graph")
    tensor1 = builder.create_const_float(1.0)
    return tensor1

t = test()  # builder 超出作用域
# ✅ tensor1 仍然有效（因为持有 builder）
# ⚠️ 但这与 C++ 的语义不同！
```

#### 影响

- **API 语义混乱**：C++ 和 Python 行为不一致
- **用户困惑**：熟悉 C++ 的用户可能误解 Python 行为
- **文档负担**：需要额外说明差异

#### 是否真的是问题？

**此做法无问题**
Python 和 C++ 的生命周期管理本就不同：
- Python: 引用计数 + GC
- C++: RAII + 手动管理

**Pythonic 的做法**：对象只要被引用就应该有效

---

### 问题 3: 多 Builder 场景的限制

#### 场景：跨 Builder 使用 Tensor

```python
builder1 = GraphBuilder("graph1")
builder2 = GraphBuilder("graph2")

tensor1 = builder1.create_const_float(1.0)

# ❌ 理论上不应该允许
result = builder2_some_op(tensor1)  # tensor1 属于 builder1

# 但由于 tensor1._builder 是 builder1
# 新生成的 tensor 也会关联到 builder1
# 导致逻辑混乱
```

#### 影响

- 跨 Builder 操作可能导致底层图结构混乱
- 难以检测和报错
- 可能导致 C++ 层断言失败

#### 解决方案

```python
# 在操作时检查 builder 一致性
def add(self, other: 'TensorHolder') -> 'TensorHolder':
    if not isinstance(other, TensorHolder):
        raise TypeError("Operand must be a TensorHolder")
    
    # 检查是否来自同一个 builder
    if self._builder is not other._builder:
        raise ValueError("Cannot operate on tensors from different GraphBuilders")
    
    # ... 后续逻辑
```

**当前代码状态**：已经添加检查

---

### 问题 4: build_and_reset() 后的状态管理

#### 场景：Builder 构建后继续使用

```python
builder = GraphBuilder("my_graph")
tensor1 = builder.create_const_float(1.0)
builder.set_graph_output(tensor1, 0)

graph = builder.build_and_reset()  # 构建完成

# ⚠️ 能否继续使用 builder？
tensor2 = builder.create_const_float(2.0)  # 不允许

# ⚠️ 能否继续使用 tensor1？
result = tensor1 + tensor2  # tensor1 的 builder已经为build过的状态
```

#### C++ 层的实现

```cpp
std::unique_ptr<ge::Graph> BuildGraphAndReset() {
    // ...
    return std::move(graph_);  // 图对象被转移！
}
```

**问题**：`build_and_reset()` 后 `graph_` 变成 nullptr，GraphBuilder 变成"空壳"

#### 影响

- `build_and_reset()` 后 builder 的状态不明确
- 继续使用可能导致未定义行为
- 旧的 tensor 引用的 builder 已经"失效"

#### 解决方案

```python
class GraphBuilder:
    def __init__(self):
        self._is_built = False
    
    def build_and_reset(self):
        if self._is_built:
            raise RuntimeError("GraphBuilder has already been built")
        
        graph_ptr = esb_lib.EsBuildGraphAndReset(self._handle)
        self._is_built = True  # 标记为已构建
        return Graph._create_from(graph_ptr)
    
    def create_const_float(self, value):
        if self._is_built:
            raise RuntimeError("Cannot create tensors after graph has been built")
        # ...
```

**当前代码状态**：已经添加检查

---

### 问题 5: Graph 对象的所有权管理冲突

#### 场景描述

Python 的 `Graph` 对象与 C++ 的 `ge::Graph*` 在不同使用场景下有**相反的所有权语义**，导致资源管理冲突。

#### 两种使用场景的所有权矛盾

**场景1：GraphBuilder.build_and_reset() 返回**

```python
builder = GraphBuilder("my_graph")
x = builder.create_input(0)
builder.set_graph_output(x, 0)
graph = builder.build_and_reset()
# 此时：Python 的 graph 对象拥有 C++ Graph* 的所有权
```

C++ 侧实现：
```cpp
EsCGraph *EsBuildGraphAndReset(EsCGraphBuilder *builder) {
    return static_cast<EsCGraph *>(
        static_cast<void *>(builder->BuildGraphAndReset().release())  // release() 转移所有权
    );
}
```

**所有权**：Python 拥有，Python 负责释放 ✅

---

**场景2：Graph 作为子图参数传入**

```python
sub_graph = create_subgraph()  # Python 拥有所有权

main_builder = GraphBuilder()
result = If(condition=..., then_graph=sub_graph, ...)
# 问题：sub_graph 的 C++ 资源被 C++ 侧接管
```

C++ 侧实现：
```cpp
Esphony_IfOutput Esphony_If(..., EsCGraph *then_branch, ...) {
    auto &builder = ...->GetOwnerBuilder();
    
    // AddResource 接管 then_branch 的所有权
    auto then_ptr = builder.AddResource(
        std::unique_ptr<ge::Graph>(then_branch)  // C++ 接管所有权
    );
    // ...
}
```

**所有权**：C++ 拥有，C++ 负责释放

**冲突**：如果 Python 也尝试释放 → 双重释放！

#### 具体问题

**问题 5.1：双重释放 (Double Free)**

```python
branch_graph = create_subgraph()
result = If(..., then_graph=branch_graph, ...)

# 问题：
# 1. C++ 侧通过 AddResource 接管了 branch_graph 的所有权
# 2. Python 的 branch_graph.__del__() 也会调用 DestroyGraph()
# → 双重释放！
```

**问题 5.2：子图 Python 对象变成悬空引用**

```python
sub_graph = create_subgraph()

main_builder = GraphBuilder()
result = If(..., then_graph=sub_graph, ...)
# sub_graph 所有权已转移

final_graph = main_builder.build_and_reset()
del main_builder  # 显示del或者脱离作用域main_builder 被 GC

#  sub_graph._handle 指向已释放的内存
print(sub_graph.name)  # 访问野指针！
```

#### 问题根源

**所有权语义的场景依赖性**：

| 场景 | Python 应该释放？ | C++ 应该释放？ | 期望行为 |
|------|-----------------|--------------|---------|
| build_and_reset() 返回 | ✅ 是 | ❌ 否 | Python 独自拥有 |
| 作为子图传入 | ❌ 否 | ✅ 是 | C++ 独自拥有 |

但 `Graph` 类的 `__del__()` 无法区分这两种场景！

```python
class Graph:
    def __del__(self):
        # ❌ 问题：不知道是哪种场景
        # 场景1：应该释放
        # 场景2：不应该释放
        destroy_graph(self._handle)  # 可能导致双重释放！
```

#### 解决方案

**引入所有权标记机制**

```python
class Graph:
    def __init__(self, name="graph"):
        self._handle = create_graph(...)
        self._owns_handle = True   # ✅ 所有权标记
        self._owner = None         # ✅ 所有权接管者引用
    
    def __del__(self):
        # ✅ 根据所有权标记决定是否释放
        if self._owns_handle:
            destroy_graph(self._handle)
    
    def _transfer_ownership_when_pass_as_subgraph(self, new_owner: GraphBuilder):
        """转移所有权到 C++ 侧
        
        Args:
            new_owner: 接管所有权的 GraphBuilder，
                      保持引用以防止其被提前 GC
        """
        self._owns_handle = False    # Python 不再释放
        self._owner = new_owner      # 保持引用，防止 new_owner 被 GC
```

**自动化处理：代码生成器插入所有权转移**

```cpp
// py_generator_utils.h - GenSubgraphConversion()
static void GenSubgraphConversion(...) {
    // 生成：subgraph._transfer_ownership_when_pass_as_subgraph(owner_graph_builder)
    ss << subgraph_name << "._transfer_ownership_when_pass_as_subgraph(" 
       << "owner_graph_builder" << ")\n";
}
```

生成的 Python 代码：
```python
def If(..., then_graph, else_graph, ...):
    owner_graph_builder = ...
    
    # ✅ 自动生成：转移所有权
    then_graph._transfer_ownership_when_pass_as_subgraph(owner_graph_builder)
    else_graph._transfer_ownership_when_pass_as_subgraph(owner_graph_builder)
    
    result = c_lib.EsphonyIf(...)
    return result
```

#### 问题解决效果

**问题 5.1 - 双重释放**：✅ 已解决
- `_transfer_ownership_when_pass_as_subgraph()` 设置 `_owns_handle=False`
- Python 的 `__del__()` 不再释放资源
- 只有 C++ 侧释放

**问题 5.2 - 子图悬空引用**：✅ 已解决
- `sub_graph._owner` 持有 `main_builder` 的引用
- 只要 `sub_graph` 存在，`main_builder` 就不会被 GC
- 只要 `main_builder` 存在，C++ 资源就有效

#### 引用链保护机制

```python
sub_graph = create_subgraph()
result = If(..., then_graph=sub_graph, ...)
```
```
Python 引用链:

    sub_graph (Graph)
        │
        └─ _owner ──────────┐
                            ↓
    result (TensorHolder)   main_builder (GraphBuilder)
        │                       │
        └─ _builder ────────────┘
                                └─ _handle → EsCGraphBuilder
                                              └─ resource_holder_
                                                   └─ [sub_graph 的 C++ Graph*]

保证：
1. result 存在 → main_builder 存在 → C++ 资源有效
2. sub_graph 存在 → main_builder 存在 → C++ 资源有效
```

**当前代码状态**：✅ 已实现并测试

---





