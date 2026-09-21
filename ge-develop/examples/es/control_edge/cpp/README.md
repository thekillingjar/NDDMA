# 样例使用指导

## 1、功能描述
本样例使用控制边进行构图，旨在帮助构图开发者快速理解控制边的概念和使用控制边进行构图

## 2、目录结构
```angular2html
cpp/
├── src/
|   └── CMakeLists.txt                // CMake构建文件
|   └── es_showcase.h                 // 头文件
|   └── make_control_edge_graph.cpp   // sample文件
├── CMakeLists.txt                    // CMake构建文件
├── main.cpp                          // 程序主入口
├── README.md                         // README文件
├── run_sample.sh                     // 执行脚本
├── utils.h                           // 工具文件
```
## 3、使用方法

### 3.1、准备cann包
- 通过安装指导 [环境准备](../../../../docs/build.md#2-安装软件包)正确安装`toolkit`和`ops`包
- 设置环境变量 (假设包安装在/usr/local/Ascend/)
```
source /usr/local/Ascend/cann/set_env.sh 
```
### 3.2、编译和执行

只需运行下述命令即可完成清理、生成接口、构图和DUMP图：
```bash
bash run_sample.sh
```
当前 run_sample.sh 的行为是：先自动清理旧的 build，构建 sample并默认执行sample dump 。当看到如下信息，代表执行成功：
```
[Success] sample 执行成功，pbtxt dump 已生成在当前目录。该文件以 ge_onnx_ 开头，可以在 netron 中打开显示
```
#### 输出文件说明

执行成功后会在当前目录生成以下文件：
- `ge_onnx_*.pbtxt` - 图结构的protobuf文本格式，可用netron查看

#### 构建图并执行

除了基本的图构建和dump功能外，esb_sample还支持构建图并实际执行计算。

```bash
bash run_sample.sh -t sample_and_run
```
该命令会：
1. 自动生成ES接口
2. 编译sample程序
3. 生成dump图、运行图并输出计算结果

执行成功后会看到：
```
[Success] sample_and_run 执行成功，pbtxt和data输出dump 已生成在当前目录
```
可通过data文件查看计算结果
### 3.3、日志打印
可执行程序执行过程中如果需要日志打印来辅助定位，可以在bash run_sample.sh之前设置如下环境变量来让日志打印到屏幕
```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1 #日志打印到屏幕
export ASCEND_GLOBAL_LOG_LEVEL=0 #日志级别为debug级别
```
### 3.4、图编译流程中DUMP图
可执行程序执行过程中，如果需要DUMP图来辅助定位图编译流程，可以在 bash run_sample.sh -t sample_and_run 之前设置如下环境变量来DUMP图到执行路径下
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
- ES API 提供了 `AddControlEdge()` 方法，支持在 C++、C使用
- 可以为一个目标节点添加多个源节点的控制依赖

## 5、控制依赖关系示例

本文档展示如何在 C 和 C++ 层面表达控制依赖关系。

### 5.1、场景说明

假设我们有以下计算需求：

```
节点1: tensor_a = Const(1.0)
节点2: tensor_b = Const(2.0)
节点3: tensor_c = Add(tensor_a, tensor_b)

控制依赖: tensor_c 必须在 tensor_a 和 tensor_b 之后执行
```

虽然 `Add` 操作已经有数据依赖(依赖 tensor_a 和 tensor_b)，但这里我们用它来演示控制依赖的表达方式。

### 5.2、C 层 API 示例

```c
#include "esb_funcs.h"

// 1. 创建图构建器
EsCGraphBuilderPtr builder = EsCreateGraphBuilder("control_dep_example");

// 2. 创建节点
EsCTensorHolderPtr tensor_a = EsCreateScalarFloat(builder, 1.0f);
EsCTensorHolderPtr tensor_b = EsCreateScalarFloat(builder, 2.0f);

// 3. 创建依赖目标节点（假设有 EsAdd 函数）
EsCTensorHolderPtr tensor_c = EsAdd(tensor_a, tensor_b);

// 4. 添加控制依赖：tensor_c 依赖于 tensor_a 和 tensor_b
EsCTensorHolderPtr src_tensors[] = {tensor_a, tensor_b};
uint32_t ret = EsAddControlEdge(tensor_c, src_tensors, 2);

if (ret != 0) {
    printf("Failed to add control dependency\n");
    // 错误处理
}

// 5. 设置输出并构建图
EsSetGraphOutput(tensor_c, 0);
EsCGraphPtr graph = EsBuildGraph(builder);

// 6. 清理
EsDestroyGraphBuilder(builder);
```
---

### 5.3、C++ 层 API 示例

```cpp
# include "es_graph_builder.h"

using namespace ge::es; 
void build_graph_with_control_dep() { 
  // 1. 创建图构建器
  EsGraphBuilder builder("control_dep_example");

  // 2. 创建节点 
  auto tensor_a = builder.CreateScalar(1.0f);
  auto tensor_b = builder.CreateScalar(2.0f);

  // 3. 创建依赖目标节点 && 添加控制依赖
  auto tensor_c = Add(tensor_a, tensor_b)
  (void) tensor_c.AddControlEdge({tensor_a, tensor_b});

  // 5. 设置输出并构建
  builder.SetOutput(tensor_c, 0);
  auto graph_ptr = builder.build_and_reset();
}

```