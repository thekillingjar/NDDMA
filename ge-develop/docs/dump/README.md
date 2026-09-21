# GE 图 Dump 格式说明

## 概述

GE（Graph Engine）支持将计算图导出为多种格式，便于开发者查看、调试和分析图结构。本文档介绍三种 dump 格式：`ge_proto`、`onnx` 和 `readable`，以及它们的特点和使用方法。

---

## Dump 格式概览

| 格式 | 文件命名 | 主要特点                                                                           |
|------|---------|--------------------------------------------------------------------------------|
| **ge_proto** | `ge_proto*.txt` | protobuf文本格式，**信息完整性最好**，可以转成JSON格式文件方便用户定位问题                       |
| **onnx** | `ge_onnx*.pbtxt` | 基于ONNX的模型描述结构，支持 [Netron](https://netron.app/) 等**可视化**工具打开。详细说明见[Netron 可视化说明](#netron-可视化说明) |
| **readable** | `ge_readable*.txt` | 类似Dynamo fx图风格，**文本可读性最高**。详细格式说明请参考 [readable_dump.md](./readable_dump.md) |

---

## Dump 使用方式

### 通过环境变量自动 Dump

通过设置环境变量，可以在图执行时自动生成 dump 文件：

```bash
# 设置图 dump 级别
export DUMP_GE_GRAPH=1

# 设置 dump 路径
export DUMP_GRAPH_PATH=/path/to/dump/directory

# 设置 dump 格式
export DUMP_GRAPH_FORMAT="ge_proto|onnx|readable"
```

**环境变量说明：**

| 环境变量 | 说明                                                                                                                                                                                                                                                    | 示例值 |
|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| `DUMP_GE_GRAPH` | 控制图 dump 的内容粒度：<br>- `1`：包含连边关系和数据信息的全量 dump<br>- `2`：不含有权重等数据的基本版 dump<br>- `3`：只显示节点关系的精简版 dump                                                                                                                                                     | `1`、`2` 或 `3` |
| `DUMP_GRAPH_PATH` | dump 文件保存路径：<br>- 可配置为绝对路径或脚本执行目录的相对路径<br>- 路径支持大小写字母、数字、下划线、中划线、句点、中文字符                                                                                                                                                                              | `/path/to/dump` |
| `DUMP_GRAPH_FORMAT` | dump 格式，支持 `ge_proto`、`onnx`、`readable`，多个格式用 `\|` 分隔                                                                                                                                                                                                 | `readable` 或 `ge_proto\|onnx`（默认值） |
| `DUMP_GRAPH_LEVEL` | 控制 dump 图编译阶段的个数：<br>- **数值配置**：<br>  - `1`：dump 所有阶段的图<br>  - `2`：dump 白名单阶段的图（默认值）<br>  - `3`：dump 最后的生成图（经过 GE 优化、编译后的图）<br>  - `4`：dump 最早的生成图（GE 解析映射算子后的编译入口图）<br>- **字符串配置**：用 `\|` 分隔，例如 `"PreRunBegin\|AfterInfershape"`，表示 dump 名称包含这些字符串的图 | `1`、`2`、`3`、`4` 或 `"PreRunBegin\|AfterInfershape"` |

### 通过 Graph API 导出

#### C++

```cpp
#include "ge/graph.h"

// 创建图
ge::Graph graph("my_graph");
// ... 构建图结构 ...

// 导出为不同格式
graph.DumpToFile(ge::Graph::DumpFormat::kTxt, "suffix");        // ge_proto
graph.DumpToFile(ge::Graph::DumpFormat::kOnnx, "suffix");       // onnx
graph.DumpToFile(ge::Graph::DumpFormat::kReadable, "suffix");   // readable
```

#### Python

```python
from ge.graph import Graph, DumpFormat

# 创建图
graph = Graph("my_graph")
# ... 构建图结构 ...

# 方式1: 导出为文件
graph.dump_to_file(format=DumpFormat.kTxt, suffix="suffix")        # ge_proto
graph.dump_to_file(format=DumpFormat.kOnnx, suffix="suffix")       # onnx
graph.dump_to_file(format=DumpFormat.kReadable, suffix="suffix")   # readable

# 方式2: 直接打印（仅 readable 格式支持）
print(graph)  # 直接打印 readable 格式到控制台查看图结构
readable_str = str(graph)  # 获取 readable 格式字符串，可用于保存或进一步处理
```
> 关于 `ge.graph` 的详细说明，请参考 [graph模块](../ge_python/design/ge_python.md#graph模块)

---

## 附录

### Netron 可视化说明

在 Netron 中打开 `ge_onnx*.pbtxt` 文件时：

- **节点表示**：图中的每个节点表示为一个算子
- **连边关系**：连边关系用带箭头的实线表示，箭头方向表示数据流向（从源节点指向目标节点）
- **节点信息查看**：点击算子节点可查看算子的详细信息，重点信息包括：

    | 属性名               | 说明                |
    | -------------------- | ------------------- |
    | `type`               | 算子类型            |
    | `name`               | 算子名              |
    | `input_desc_dtype:x` | 第x个输入的数据类型 |
    | `input_desc_layout:x` | 第x个输入的数据格式 |
    | `input_desc_shape:x` | 第x个输入的shape    |
    | `output_desc_dtype:x` | 第x个输出的数据类型 |
    | `output_desc_layout:x` | 第x个输出的数据格式 |
    | `output_desc_shape:x` | 第x个输出的shape    |
