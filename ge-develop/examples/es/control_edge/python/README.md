# 样例使用指导

## 1、功能描述
本样例使用控制边进行构图，旨在帮助构图开发者快速理解控制边的概念和使用控制边进行构图

## 2、目录结构
```angular2html
python/
├── src/
|   └── make_control_edge_graph.py // sample文件
├── CMakeLists.txt                 // 编译脚本
├── README.md                      // README文件
├── run_sample.sh                  // 执行脚本
```

## 3、使用方法

### 3.1、准备cann包
- 通过安装指导 [环境准备](../../../../docs/build.md#2-安装软件包)正确安装`toolkit`和`ops`包
- 设置环境变量 (假设包安装在/usr/local/Ascend/)
```
source /usr/local/Ascend/cann/set_env.sh 
```
### 3.2、编译和执行
- 注：和 C/C++构图对比，Python构图需要额外添加 LD_LIBRARY_PATH 和 PYTHONPATH(参考sample中的配置方式)

```bash
bash run_sample.sh -t sample_and_run_python
```
该命令会：
1. 自动生成ES接口
2. 编译sample程序
3. 生成dump图并运行该图

执行成功后会看到：
```
[Success] sample 执行成功，pbtxt dump 已生成在当前目录。该文件以 ge_onnx_ 开头，可以在 netron 中打开显示
```
#### 输出文件说明

执行成功后会在当前目录生成以下文件：
- `ge_onnx_*.pbtxt` - 图结构的protobuf文本格式，可用netron查看

### 3.3、日志打印
可执行程序执行过程中如果需要日志打印来辅助定位，可以在bash run_sample.sh -t sample_and_run_python之前设置如下环境变量来让日志打印到屏幕
```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1 #日志打印到屏幕
export ASCEND_GLOBAL_LOG_LEVEL=0 #日志级别为debug级别
```
### 3.4、图编译流程中DUMP图
可执行程序执行过程中，如果需要DUMP图来辅助定位图编译流程，可以在 bash run_sample.sh -t sample_and_run_python 之前设置如下环境变量来DUMP图到执行路径下
```bash
export DUMP_GE_GRAPH=2 
```

## 4、核心概念介绍

### 4.1、构图步骤如下：
- 创建图构建器(用于提供构图所需的上下文、工作空间及构建相关方法)
- 添加起始节点(起始节点指无输入依赖的节点，通常包括图的输入(如 Data 节点)和权重常量(如 Const 节点))
- 添加中间节点(中间节点为具有输入依赖的计算节点，通常由用户构图逻辑生成，并通过已有节点作为输入连接)
- 设置图输出(明确图的输出节点，作为计算结果的终点)

### 4.2、控制边（Control Edge）
**概念说明：**
控制边用于在计算图中指定节点的执行顺序，即使这些节点之间没有数据依赖关系。控制边不传递数据，只传递控制信号，确保源节点在目标节点之前执行。

**构图 API 特点：**
- ES API 提供了 `add_control_dependency()` 方法，支持在 Python中使用
- 可以为一个目标节点添加多个源节点的控制依赖

## 5、控制依赖关系示例

### 5.1、概述

控制依赖（Control Dependency）用于在计算图中指定节点的执行顺序，即使这些节点之间没有数据依赖。本文档展示如何在 Python 层面表达控制依赖关系。

## Python 层 API 示例

### 方式 1: 直接调用 add_control_dependency()

```python
from ge.es import GraphBuilder

# 1. 创建图构建器
builder = GraphBuilder("control_dep_example")

# 2. 创建节点
tensor_a = builder.create_scalar_float(1.0)
tensor_b = builder.create_scalar_float(2.0)

# 3. 创建依赖目标节点（假设有生成的 Add 操作）
from ge.es.all import Add

tensor_c = Add(tensor_a, tensor_b)

# 4. 添加控制依赖：tensor_c 依赖于 tensor_a 和 tensor_b
builder.add_control_dependency(
    dst_tensor=tensor_c,
    src_tensors=[tensor_a, tensor_b]
)

# 5. 设置输出并构建
builder.set_graph_output(tensor_c, 0)
graph = builder.build_and_reset()
```

---

### 方式 2: 使用 control_dependency_scope 上下文管理器（推荐）

```python
from ge.es import GraphBuilder
from ge.es.graph_builder import control_dependency_scope
from ge.es.all import Add

# 1. 创建图构建器
builder = GraphBuilder("control_dep_scope_example")

# 2. 创建依赖源节点
tensor_a = builder.create_scalar_float(1.0)
tensor_b = builder.create_scalar_float(2.0)

# 3. 使用 scope：在 scope 内创建的所有节点自动依赖 tensor_a 和 tensor_b
with control_dependency_scope([tensor_a, tensor_b]):
    # 在此 scope 内创建的节点会自动添加控制依赖
    tensor_c = builder.create_scalar_float(3.0)
    tensor_d = Add(tensor_a, tensor_c)

    # tensor_c 和 tensor_d 的生产节点都自动依赖 tensor_a 和 tensor_b

# 4. 设置输出并构建
builder.set_graph_output(tensor_d, 0)
graph = builder.build_and_reset()
```

#### 嵌套 scope 示例

```python
from ge.es import GraphBuilder
from ge.es.graph_builder import control_dependency_scope
from ge.es.all import Add


def build_graph_with_nested_scopes():
    """演示嵌套的控制依赖 scope"""
    builder = GraphBuilder("nested_scopes")

    # 全局初始化节点
    global_init = builder.create_scalar_float(0.0)

    # 第一层 scope
    with control_dependency_scope([global_init]):
        # 模块 A 的初始化
        module_a_init = builder.create_scalar_float(1.0)

        # 第二层嵌套 scope
        with control_dependency_scope([module_a_init]):
            # 模块 A 的计算（依赖 global_init 和 module_a_init）
            module_a_output = Add(module_a_init, global_init)

        # 返回第一层 scope
        # 模块 B 的初始化（只依赖 global_init）
        module_b_init = builder.create_scalar_float(2.0)

    # 最终输出（不在任何 scope 中，没有额外的控制依赖）
    final_output = Add(module_a_output, module_b_init)

    builder.set_graph_output(final_output, 0)
    return builder.build_and_reset()


# 使用
graph = build_graph_with_nested_scopes()
```