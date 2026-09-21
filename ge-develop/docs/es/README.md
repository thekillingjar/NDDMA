# ES (Eager Style)文档

## 概述

ES (Eager Style)是 GraphEngine 中用于构建计算图的函数式接口模块，提供了便捷的图构建功能。ES
模块支持多种编程语言（C、C++、Python），为用户提供了灵活且易用的图构建方式。

**ES 系列 API 核心特点：**

* **自动生成**：API 并非手写，而是基于算子原型定义自动生成，减少算子开发者负担，特别是自定义算子也需要使用ES的API的构图场景。
* **多语言支持**：原生支持 C、C++，并可扩展支持 Python，满足不同开发习惯。
* **全维度兼容**：通过良好的API设计和代码生成机制，结合IR语义兼容处理，从源头实现了前后向API,ABI的兼容性保障。

**ES 三大关键组件：**

1. **ES 基础数据结构**（GE仓/GE包）：提供 `EsGraphBuilder`（图构建器）、`EsTensorHolder`（张量持有者）等核心基础设施，是构图的基石。
2. **ES Code Generator**（GE仓/GE包）：核心工具 `gen_esb`，负责读取算子原型定义，自动化生成每个算子对应的构图 API
   代码。
3. **Generated ES API**（OPP包）：由生成器构建的最终产物，包含在算子包中，是用户直接调用的函数接口。

## 快速导航

### API 参考文档

- **[Python API](api/es_python.md)** - Eager Style Graph Builder Python 接口文档
  - GraphBuilder、TensorHolder 等基础类的主要接口
- **[C++/C API](api/es_cpp.md)** - Eager Style Graph Builder C++/C 接口文档
  - EsGraphBuilder、EsTensorHolder 等基础类的主要接口

### 设计文档

- **[整体架构设计](design/architecture_design.md)** - ES模块整体架构设计说明
- **[所有权关系分析](design/ownership_analysis.md)** - ES模块Python 层和 C++ 层资源管理机制说明

### 工具文档

- **[gen_esb 代码生成工具](tools/gen_esb.md)** - ES 代码生成工具使用说明
  - 工具功能说明
  - 使用方法和参数说明

- **[generate_es_package.cmake](tools/generate_es_package_cmake_readme.md)** - ES 代码生成、构建、安装一键式
  cmake 函数
  - 函数功能说明
  - 使用方法和参数说明

### RFC 文档

- **RFC 目录** - 设计提案文档
  - 新功能提案
  - 架构改进方案
  - API 变更建议

## 文档结构

```
docs/es/
├── README.md                    # 本文件，文档导航入口
├── api/                         # API 参考文档
├── design/                      # 设计和技术分析文档
├── tools/                       # 工具使用文档
└── rfc/                         # RFC 提案文档
```

## sample

- [使用es构图的多语言sample](../../examples/es)

- [自定义es的api构图的多语言sample](../../examples/custom_es_api)


## 开发路线图

我们在2025年首次推出了ES的功能，目标是提供便捷的多语言图构建能力，支持全量算子的自动API生成，并实现良好的前后向兼容性。2026年Q1阶段我们将重点完成API集成和C++构图场景的向后兼容能力，让用户可以直接安装ops包后使用ES构图能力，无需手动生成代码。具体的研发Roadmap如下：

---

### 核心架构

- [x] [***December 2025***]ES核心架构设计完成并落地，支持全量原型的代码生成和多语言(C, C++, Python)构图。

### API 集成

- [x] [***December 2025***]math 分包已完成ES API集成。
- [x] [***February 2026***]nn、cv、transformer完成ES API集成。
- [ ] [***April 2026***]hcom分包完成ES API集成。

注：在全量原型的ES API打包到各个算子分包之前，如需使用 ES API，可以参考工具文档中的 [generate_es_package.cmake](tools/generate_es_package_cmake_readme.md) 进行代码生成和集成。

### sample和相关文档

- [x] [***December 2025***]已提供多语言的 sample，涵盖常见使用场景。
- [x] [***December 2025***]已提供细化的文档，即本目录。
- [x] [***January 2026***]集合通信的样例完成交付。

### 后向兼容
- [x] [***December 2025***]Python API后向兼容完成设计并落地。
- [x] [***January 2026***]C++ API后向兼容性设计完成；C++ API的后向兼容依赖完成`历史原型库`的设计方案。
- [x] [***March 2026***]C++ API后向兼容性代码开发完成；实现C++构图场景的完整向后兼容能力。

### 自定义算子工程集成

- [ ] [***TBD***]ascendc 自定义算子工程集成 ES 的代码生成能力，暂无规划。


