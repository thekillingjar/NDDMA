# Readable Dump 文档

## 概述

Readable Dump 是 GE（Graph Engine）提供的高可读性图结构导出格式，采用类似 Dynamo fx 图风格的文本表示，便于开发者直接阅读和理解计算图结构。

## 设计原则

**可读性为最高优先级，简单优于完整**：

- **展示方式**：通过类似函数调用的方式，展示 IR 属性、输入、输出的信息
- **信息过滤**：控制边和私有属性等非 IR 定义的信息不展示（影响整体可读性）
- **数值缩略**：Const 携带的数值采用缩略展示，详见[附录：Const 常量展示规则](#const-常量展示规则)

---

## readable 文件解析

示例 1： 如下为一个完整的ge_readable*.txt文件：
```
graph("MakeTransformerSubGraph"):
  %input_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %input_1 : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %input_2 : [#users=1] = Node[type=Data] (attrs = {index: 2})
  %Const_0 : [#users=1] = Node[type=Const] (attrs = {value: [-1 7168]})
  %Reshape_1 : [#users=1] = Node[type=Reshape] (inputs = (x=%input_0, shape=%Const_0), attrs = {axis: 0, num_axes: -1})
  %Cast_2 : [#users=1] = Node[type=Cast] (inputs = (x=%Reshape_1), attrs = {dst_type: 0})
  %Cast_3 : [#users=1] = Node[type=Cast] (inputs = (x=%input_1), attrs = {dst_type: 0})
  %Const_4 : [#users=1] = Node[type=Const] (attrs = {value: [1 0]})
  %Transpose_5 : [#users=1] = Node[type=Transpose] (inputs = (x=%Cast_3, perm=%Const_4))
  %MatMul_6 : [#users=1] = Node[type=MatMul] (inputs = (x1=%Cast_2, x2=%Transpose_5), attrs = {transpose_x1: false, transpose_x2: false})
  %Sigmoid_7 : [#users=1] = Node[type=Sigmoid] (inputs = (x=%MatMul_6))
  %Const_8 : [#users=1] = Node[type=Const] (attrs = {value: [-1 256]})
  %Reshape_9 : [#users=1] = Node[type=Reshape] (inputs = (x=%Sigmoid_7, shape=%Const_8), attrs = {axis: 0, num_axes: -1})
  %Unsqueeze_10 : [#users=1] = Node[type=Unsqueeze] (inputs = (x=%input_2), attrs = {axes: {0}})
  %Cast_11 : [#users=1] = Node[type=Cast] (inputs = (x=%Unsqueeze_10), attrs = {dst_type: 0})
  %Add_12 : [#users=1] = Node[type=Add] (inputs = (x1=%Reshape_9, x2=%Cast_11))
  %Const_13 : [#users=1] = Node[type=Const] (attrs = {value: [2]})
  %TopKV2_14 : [#users=2] = Node[type=TopKV2] (inputs = (x=%Add_12, k=%Const_13), attrs = {sorted: true, dim: -1, largest: true, indices_dtype: 3})
  %ret : [users=1] = get_element[node=%TopKV2_14](0)
  %ret_1 : [users=0] = get_element[node=%TopKV2_14](1)
  %Const_15 : [#users=1] = Node[type=Const] (attrs = {value: [-1]})
  %ReduceSum_16 : [#users=1] = Node[type=ReduceSum] (inputs = (x=%ret, axes=%Const_15), attrs = {keep_dims: false, noop_with_empty_axes: true})
  %Const_17 : [#users=1] = Node[type=Const] (attrs = {value: [4]})
  %TopKV2_18 : [#users=2] = Node[type=TopKV2] (inputs = (x=%ReduceSum_16, k=%Const_17), attrs = {sorted: false, dim: -1, largest: true, indices_dtype: 3})
  %ret_2 : [users=0] = get_element[node=%TopKV2_18](0)
  %ret_3 : [users=1] = get_element[node=%TopKV2_18](1)
  %Cast_19 : [#users=1] = Node[type=Cast] (inputs = (x=%ret_3), attrs = {dst_type: 9})
  %ZerosLike_20 : [#users=1] = Node[type=ZerosLike] (inputs = (x=%ReduceSum_16))
  %Shape_21 : [#users=1] = Node[type=Shape] (inputs = (x=%Cast_19), attrs = {dtype: 3})
  %Const_22 : [#users=1] = Node[type=Const] (attrs = {value: [1.000000]})
  %Cast_23 : [#users=1] = Node[type=Cast] (inputs = (x=%Const_22), attrs = {dst_type: 0})
  %Fill_24 : [#users=1] = Node[type=Fill] (inputs = (dims=%Shape_21, value=%Cast_23))
  %ScatterElements_25 : [#users=1] = Node[type=ScatterElements] (inputs = (data=%ZerosLike_20, indices=%Cast_19, updates=%Fill_24), attrs = {axis: 0, reduction: "none"})
  %Unsqueeze_26 : [#users=1] = Node[type=Unsqueeze] (inputs = (x=%ScatterElements_25), attrs = {axes: {-1}})
  %Const_27 : [#users=1] = Node[type=Const] (attrs = {value: [256 256]})
  %BroadcastTo_28 : [#users=1] = Node[type=BroadcastTo] (inputs = (x=%Unsqueeze_26, shape=%Const_27))
  %Identity_29 : [#users=1] = Node[type=Identity] (inputs = (x=%BroadcastTo_28))
  %Const_30 : [#users=1] = Node[type=Const] (attrs = {value: [256 256]})
  %Reshape_31 : [#users=1] = Node[type=Reshape] (inputs = (x=%Identity_29, shape=%Const_30), attrs = {axis: 0, num_axes: -1})
  %Cast_32 : [#users=1] = Node[type=Cast] (inputs = (x=%Reshape_31), attrs = {dst_type: 12})
  %LogicalNot_33 : [#users=1] = Node[type=LogicalNot] (inputs = (x=%Cast_32))
  %Const_34 : [#users=1] = Node[type=Const] (attrs = {value: [0.000000]})
  %MaskedFill_35 : [#users=1] = Node[type=MaskedFill] (inputs = (x=%Add_12, mask=%LogicalNot_33, value=%Const_34))
  %Const_36 : [#users=1] = Node[type=Const] (attrs = {value: [4]})
  %TopKV2_37 : [#users=2] = Node[type=TopKV2] (inputs = (x=%MaskedFill_35, k=%Const_36), attrs = {sorted: false, dim: -1, largest: true, indices_dtype: 3})
  %ret_4 : [users=0] = get_element[node=%TopKV2_37](0)
  %ret_5 : [users=1] = get_element[node=%TopKV2_37](1)
  %Cast_38 : [#users=1] = Node[type=Cast] (inputs = (x=%ret_5), attrs = {dst_type: 9})
  %GatherElements_39 : [#users=1] = Node[type=GatherElements] (inputs = (x=%Sigmoid_7, index=%Cast_38), attrs = {dim: 1})
  %Const_40 : [#users=1] = Node[type=Const] (attrs = {value: [0.000001]})
  %RealDiv_41 : [#users=1] = Node[type=RealDiv] (inputs = (x1=%GatherElements_39, x2=%Const_40))
  %Const_42 : [#users=1] = Node[type=Const] (attrs = {value: [2.500000]})
  %Mul_43 : [#users=1] = Node[type=Mul] (inputs = (x1=%RealDiv_41, x2=%Const_42))
  %Cast_44 : [#users=1] = Node[type=Cast] (inputs = (x=%Mul_43), attrs = {dst_type: 0})

  return (output_0=%Cast_38, output_1=%Cast_44)
```

示例 2：如下为一个包含子图的文件：
```
graph("TransformerBlockSubgraph"):
  %input_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %pred_1 : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %If_0 : [#users=1] = Node[type=If] (inputs = (cond=%pred_1, input_0=%input_0), attrs = {then_branch: %If_then, else_branch: %If_else})
  %Const_1 : [#users=1] = Node[type=Const] (attrs = {value: [0]})
  %Const_2 : [#users=1] = Node[type=Const] (attrs = {value: [4]})
  %Const_3 : [#users=1] = Node[type=Const] (attrs = {value: [1]})
  %For_6 : [#users=2] = Node[type=For] (inputs = (start=%Const_1, limit=%Const_2, delta=%Const_3, input_1=%If_0), attrs = {body: %For_body})
  %ret : [users=1] = get_element[node=%For_6](0)
  %ret_1 : [users=1] = get_element[node=%For_6](1)

  return (output_0=%ret, output_1=%ret_1)

graph("If_then"):
  %input_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %Const_0 : [#users=1] = Node[type=Const] (attrs = {value: [0.900000]})
  %Mul_1 : [#users=1] = Node[type=Mul] (inputs = (x1=%input_0, x2=%Const_0))
  return (%Mul_1)

graph("If_else"):
  %input_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %Identity_0 : [#users=1] = Node[type=Identity] (inputs = (x=%input_0))
  return (%Identity_0)

graph("For_body"):
  %iter : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %hidden : [#users=1] = Node[type=Data] (attrs = {index: 1})
  %Const_0 : [#users=1] = Node[type=Const] (attrs = {value: [1]})
  %Add_0 : [#users=1] = Node[type=Add] (inputs = (x1=%iter, x2=%Const_0))
  %Const_1 : [#users=1] = Node[type=Const] (attrs = {value: [0.500000]})
  %Mul_1 : [#users=1] = Node[type=Mul] (inputs = (x1=%hidden, x2=%Const_1))
  %Add_2 : [#users=1] = Node[type=Add] (inputs = (x1=%hidden, x2=%Mul_1))

  return (output_0=%Add_0, output_1=%Add_2)
```

字段解释如下：
- **图名称**：`graph("<图名称>")`。
- **节点实例**：`%<节点实例名称> : [#users=<出度>] = Node[type=<节点类型>](inputs = (<输入名称1>=%<输入实例1>, ...), attrs = {<属性名称1>: <属性值1>, ...})`。
    - `%<节点实例名称>`：节点实例名称。
    - `[#users=<出度>]` ：节点输出个数。
    - `Node[type=<节点类型>]`：节点对应的算子类型，例如 MatMul 节点显示为 `Node[type=MatMul]`。
    - `inputs = (<输入名称1>=%<输入实例1>, ...)`：
       节点输入以“参数名=实例名”形式展示；参数名解析异常时，回退为 `_input_N` 序列。当节点无输入时该字段缺省。
       - **动态输入**：对动态输入参数使用 `输入名称_#cnt` 形式编号（`输入名称_0, 输入名称_1, ...`）。
    - `attrs = {<属性名称1>: <属性值1>, ...}`：
       节点属性集合，包含**子图属性项**与普通属性项，当属性为空时该字段缺省。
- **多输出节点的输出引用**：`%ret/ret_#cnt : [#users=<消费者个数>] = get_element[node=%<节点实例名称>](<输出index>)`。
    - `%ret/ret_#cnt`：多输出节点的每一路输出统一使用ret命名 `ret, ret_1, ret_2, ...`。
    - `[#users=<消费者个数>]`：该输出的消费者数量。
    - `get_element[node=%<节点实例名称>](<输出index>)`：以 `get_element` 表示“从多输出节点提取第 `<输出index>` 路输出”。
- **图输出**：`return (<输出列表>)`  
  表示图的输出，对应 `NetOutput` 节点的输入。
    - **单输出**：`return (%<输出实例>)`。
    - **多输出**：`return (output_0=%<输出实例0>, output_1=%<输出实例1>, ...)`。输出使用 `output_#cnt` 形式编号（`output_0, output_1, ...`）。
- **子图表示**：  
  包含子图的节点按以下规则表示：
    - **子图声明**：在父节点的 `attrs` 中声明：`attrs = {<子图属性名称>: %<子图实例名>, ... }`；子图属性名称解析异常时，回退为 `_graph_N` 序列。
    - **输入对应关系**：父节点的输入 `input_#cnt`（或 `args_#cnt`）对应子图中 `index` 属性值为 `cnt` 的 `Data` 节点。  
      例如：父节点的 `input_0` → 子图的 `Data(attrs = {index: 0})`。
    - **输出对应关系**：
      - **单输出**：子图返回值直接作为父节点输出。
      - **多输出**：子图的输出 `output_#cnt` 对应父节点的第 `cnt` 路输出，父节点通过 `get_element[node=%<父节点>](cnt)` 提取对应输出。
    - **子图展示位置**：  
      子图内容在父图主体输出结束后单独展示；每个子图以 `graph("<子图实例名>"):` 开始。

---

## 附录

### Const 常量展示规则

为平衡可读性与信息完整性，Const 节点的 `value` 属性按以下规则缩略展示：

- **无 value 属性**：不展示
- **空 tensor**：表示没有数据内容，展示为 `value: <empty>`
- **单个数值**：代表标量 `[]` 或 单元素`[1]`，直接展示实际数值，如 `value: [1.5]`
- **多个数值**：当数值超过 6 个时，采用首尾各保留 3 个数值的缩略形式，如 `value: [1 2 3 ... 98 99 100]`
- **不支持的数据类型**：展示为 `value: <not_supported>`
