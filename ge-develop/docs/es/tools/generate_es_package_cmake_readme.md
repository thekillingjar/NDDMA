# add_es_library 使用指南

## 概述

提供如下 CMake 函数用于生成 Eager Style (ES)产物：

- **`add_es_library_and_whl`**: 生成 C/C++ 动态库 + Python wheel 包（完整产物）
- **`add_es_library`**: 只生成 C/C++ 动态库（纯 C/C++ 项目使用）

## 快速开始

### 前置要求
1. 通过[安装指导](../../build.md#2-安装软件包)正确安装`toolkit`包，并按照指导**正确配置环境变量**

2. 定义原型动态库
```cmake
add_library(opgraph_math SHARED #要求是so的形式
)
```

### 1. 引入函数

#### 使用 find_package

```cmake
# 在你的 CMakeLists.txt 中添加模块路径(ASCEND_HOME_PATH来自`前置要求`中的配置环境变量)
list(APPEND CMAKE_MODULE_PATH "${ASCEND_HOME_PATH}/include/ge/cmake")

# 查找模块
find_package(GenerateEsPackage REQUIRED)
```

**说明**：
- 当前版本需要在 CMakeLists.txt 中手动添加 `CMAKE_MODULE_PATH`
- **计划支持**：未来版本将支持只 `source set_env.sh` 后直接 `find_package`（无需手动添加路径）

### 2. 生成 ES API

#### 生成完整产物

```cmake
add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_math
    OPP_PROTO_TARGET  opgraph_math #`前置要求`中的原型库target
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
)
```

#### 只生成 C/C++ 库

```cmake
add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_math
    OPP_PROTO_TARGET  opgraph_math
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
)
```

### 3. 使用生成的产物

```cmake
# 在你的应用中链接
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE es_math)
```

### 4. 构建

```bash
# 直接构建你的应用即可
make my_app
# → 会自动触发 ES 包的构建
```

## 参数说明

### 函数参数

两个函数参数完全相同：

| 参数 | 必需性 | 说明 | 示例 |
|------|--------|------|------|
| `ES_LINKABLE_AND_ALL_TARGET` | ✅ 必需 | 对外暴露的库 target 名称 | `es_math`, `es_nn`, `es_cv` |
| `OPP_PROTO_TARGET` | ✅ 必需 | 算子原型库的 CMake target 名称 | `opgraph_math`, `opgraph_nn` |
| `OUTPUT_PATH` | ✅ 必需 | 产物输出的根目录 | `${CMAKE_BINARY_DIR}/output` |
| `EXCLUDE_OPS` | ☑️ 可选  | 需要排除生成的算子 | `Add,Conv2D` |

**区别**：
- `add_es_library_and_whl`: 生成库 + wheel 包
- `add_es_library`: 只生成库（跳过 wheel 包生成）

**重要**：
- 因为 ES 产物是函数内部动态生成，而 cmake 整体又是配置和构建分阶段处理，所以我们的函数内部会有重配置和构建的操作，这也是我们目前直接提供一个 interface
  类型的 `ES_LINKABLE_AND_ALL_TARGET` 的原因
- 函数会自动从 `OPP_PROTO_TARGET` 的 `LIBRARY_OUTPUT_DIRECTORY` 推导原型库路径，这是生成原型库对应的 ES 产物的基本条件

### 历史原型库相关 CMake 变量

以下变量在调用函数前通过 `set()` 设置，用于控制历史原型库功能：

| 变量 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `GE_ES_EXTRACT_HISTORY` | bool（可选） | ON 时向 gen_esb 传入 `--es_mode=extract_history`，启用**历史原型库生成模式**（归档当前原型为 JSON，供后续版本代码生成时使用）；不设置或 OFF 则 gen_esb 默认 `codegen`（仅生成 C++ API） | `set(GE_ES_EXTRACT_HISTORY ON)` |
| `GE_ES_RELEASE_VERSION` | 可选 | **当前新版本号**，归档步骤将当前原型归档为此版本；历史原型库生成模式下必须设置 | `set(GE_ES_RELEASE_VERSION "8.0.RC1")` |
| `GE_ES_RELEASE_DATE` | 可选 | 历史原型库生成模式下指定发布日期（格式 `YYYY-MM-DD`），不设置则由 gen_esb 使用当前日期 | `set(GE_ES_RELEASE_DATE "2026-02-28")` |
| `GE_ES_BRANCH_NAME` | 可选 | 历史原型库生成模式下指定发布分支名。**注意**：当设为 `master` 时，函数会忽略所有归档相关变量（`GE_ES_EXTRACT_HISTORY`/`GE_ES_RELEASE_VERSION`/`GE_ES_RELEASE_DATE`），仅走纯代码生成模式 | `set(GE_ES_BRANCH_NAME "release/8.0")` |

> **历史原型库路径自动推导**：函数内部根据 cmake 文件安装位置自动推导 `${CANN_INSTALL_PATH}/cann/opp/history_registry/<module>` 路径，路径存在且非空时自动向 gen_esb 传入 `--history_registry`，**调用方无需显式设置历史原型库路径**。
>
> **完整商发模式**：`GE_ES_EXTRACT_HISTORY=ON` 且 ops 包中存在历史原型库时，`add_es_library` 内部自动串行执行两次 gen_esb（代码生成 + 历史原型库归档&合并），**调用方无需额外处理**，一次 `add_es_library` 调用即可完成完整商发所需的全部操作（参见示例 9）。

## 输出产物

### add_es_library_and_whl 生成的产物

```
OUTPUT_PATH/
├── include/
│   └── es_math/               # 头文件目录
│       ├── es_math_ops.h      # C++ 接口聚合头文件
│       ├── es_math_ops_c.h    # C 接口聚合头文件
│       └── es_add.h ...       # 单个算子头文件(一般是有多个文件)
├── lib64/
│   ├── libes_math.so          # 动态库
│   └── libes_math.a           # 静态库
└── whl/
    └── es_math-1.0.0-py3-none-any.whl  # Python 包
```

### add_es_library 生成的产物

```
OUTPUT_PATH/
├── include/
│   └── es_math/               # 头文件目录
│       ├── es_math_ops.h      # C++ 接口聚合头文件
│       ├── es_math_ops_c.h    # C 接口聚合头文件
│       └── es_add.h ...       # 单个算子头文件(一般是有多个文件)
└── lib64/
    ├── libes_math.so          # 动态库
    └── libes_math.a           # 静态库
```

**说明**：
- **聚合的含义**：包含 es_math 下所有算子的构图 API
- **动态库与静态库**：每次生成会同时产出 `lib<name>.so` 与 `lib<name>.a`。对外接口 target（如 `es_math`）默认链接动态库；若需静态链接，可手动指定 `lib64/lib<name>.a` 或通过 `target_link_libraries(your_target PRIVATE ${OUTPUT_PATH}/lib64/libes_math.a)` 等方式链接静态库。

### 启用历史原型库相关模式后的额外产物

#### 历史原型库生成模式（`GE_ES_EXTRACT_HISTORY=ON`）

gen_esb 直接输出到 `OUTPUT_PATH`，生成以下 JSON 结构：

```
OUTPUT_PATH/
├── index.json                          # 版本索引（所有已归档版本的列表）
└── registry/
    └── <GE_ES_RELEASE_VERSION>/
        ├── metadata.json               # 版本元信息（版本号、发布日期、分支名等）
        └── operators.json              # 算子原型数据（IR 结构化描述）
```

#### 代码生成模式含历史兼容（自动检测到历史原型库时）

C++ 头文件中同一算子会出现多版本重载签名（历史版本签名 + 当前版本签名），`.so` 同时包含所有版本的实现，向前兼容旧版本调用方式。产物目录结构与标准模式相同，仅头文件内容有差异。

## 生成的 Target

### 对外使用的 Target

| Target 名称 | 用途 | 说明 |
|------------|------|------|
| `es_math` | **链接依赖** | **使用方通过此 target 链接，自动触发构建**；默认链接动态库（.so），静态库（.a）同时生成在 `lib64/` 下，需静态链接时请直接指定静态库文件。 |

## 使用示例

### 示例 1：基本用法

```cmake
# 1. 定义原型动态库(`前置要求`)
add_library(opgraph_math SHARED
)
# 2. 生成 ES API 包（库 + wheel 包）
add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_math
    OPP_PROTO_TARGET  opgraph_math
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
)

# 3. 在应用中使用
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE es_math)
```
> **提示**：如果工程中还有在编译阶段就需要 ES 头文件的 OBJECT/STATIC 中间目标，可额外添加
> `add_dependencies(<中间目标> build_es_math)`（`build_es_***` 名称会随包名变化），以确保在编译该
> 目标前已完成代码生成。最终的可执行或共享库只需 `target_link_libraries(... es_math)` 即可自动触发
> 相关依赖。

补充示例：包含对象库

```cmake
add_library(my_obj OBJECT foo.cc bar.cc)
# 保证编译对象前先生成 es_math 头文件(推荐显示添加)
add_dependencies(my_obj build_es_math)
# 传递头文件搜索路径的作用
target_link_libraries(my_obj PRIVATE es_math)

add_executable(my_app $<TARGET_OBJECTS:my_obj>)
target_link_libraries(my_app PRIVATE
        es_math
)
```

### 示例 2：只生成 C/C++ 库

```cmake
# 只生成 C/C++ 库（跳过 Python wheel 包）
add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_math
    OPP_PROTO_TARGET  opgraph_math
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
)
```
### 示例 3：排除部分算子生成

```cmake
# 生成 math TARGET
# 生成产物不会包含 es_Add_c.h; es_Add.h, es_Add.cpp, es_Add.py
add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_math
    OPP_PROTO_TARGET  opgraph_math
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
    EXCLUDE_OPS       Add
)
```

### 示例 4：多个 TARGET

```cmake
# 生成 math TARGET
add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_math
    OPP_PROTO_TARGET  opgraph_math
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
)

# 生成 nn TARGET
add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_nn
    OPP_PROTO_TARGET  opgraph_nn
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
)

# 使用多个TARGET
add_executable(my_inference_app main.cpp)
target_link_libraries(my_inference_app PRIVATE
    es_math
    es_nn
)
```

### 示例 5：C++ 代码中使用

```cpp
#include "es_math/es_math_ops.h"
#include "ge/es/graph_builder.h"

using namespace ge::es;

int main() {
    EsGraphBuilder builder("my_graph");
    
    // 创建输入
    auto input1 = builder.CreateConstFloat(1.0f);
    auto input2 = builder.CreateConstFloat(2.0f);
    
    // 使用生成的 ES API
    auto result = Add(input1, input2);
    
    auto graph = builder.build_and_reset();
    return 0;
}
```

### 示例 6：Python 中使用

```bash
# 安装 wheel 包
pip install output/whl/es_math-1.0.0-py3-none-any.whl
```

```python
# 通过 entry_point 机制自动加载的插件，使用统一的导入方式
from ge.es.math import Add
from ge.es import GraphBuilder

builder = GraphBuilder("my_graph")
input1 = builder.create_const_float(1.0)
input2 = builder.create_const_float(2.0)

result = Add(input1, input2)
graph = builder.build_and_reset()
```

**说明**：
- 插件通过 entry_point 机制自动加载，导入路径为 `ge.es.<module_name>`
- `module_name` 是从 `ES_LINKABLE_AND_ALL_TARGET` 去掉 `es_` 前缀得到的（如 `es_math` → `math`）
- 可以使用 `ge.es.list_plugins()` 查看所有已加载的插件名称
- 可以使用 `ge.es.get_plugin('math')` 检查插件是否存在（返回模块对象或 None）。

### 示例 7：历史原型库生成模式（商发首次构建，归档当前版本原型）

```cmake
# 从已安装的算子原型 .so 中提取 IR 原型，归档为 JSON 供下次商发使用
set(GE_ES_EXTRACT_HISTORY ON)
set(GE_ES_RELEASE_VERSION "8.0.RC1")
set(GE_ES_RELEASE_DATE    "2026-02-28")  # 可选，不设置则使用当前日期
set(GE_ES_BRANCH_NAME     "release/8.0") # 可选；若设为 master 则不会执行归档，仅走代码生成

add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_math
    OPP_PROTO_TARGET  opgraph_math
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
)
# 产物：output/index.json, output/registry/8.0.RC1/metadata.json, output/registry/8.0.RC1/operators.json
```

### 示例 8：代码生成模式含历史兼容（自动消费已有历史数据，生成带重载的 C++ API）

```cmake
# 函数内部自动检测 ${CANN_INSTALL_PATH}/cann/opp/history_registry/math，
# 路径存在且非空时自动生成带历史兼容重载签名的 C++ 接口，无需手动设置路径
set(GE_ES_RELEASE_VERSION  "8.0.RC2")

add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_math
    OPP_PROTO_TARGET  opgraph_math
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
)
# 产物头文件中同一算子出现多版本重载（旧签名 + 新签名），向前兼容旧版本调用
```

### 示例 9：完整商发模式（单次调用，函数内部自动完成代码生成 + 历史原型库归档&合并）

```cmake
# GE_ES_EXTRACT_HISTORY=ON + ops 包中存在历史原型库（自动检测）= 完整商发模式
# add_es_library 内部自动串行执行两次 gen_esb，调用方无需额外处理：
#   gen_esb 调用1：默认 codegen 模式，以当前日期为锚点自动选取窗口内历史版本 → 对比当前原型 → 生成带重载 C++ API
#   gen_esb 调用2：--es_mode=extract_history，将已有历史库从 _AUTO_HISTORY_REGISTRY 复制至 OUTPUT_PATH，在 OUTPUT_PATH 追加当前版本原型 → OUTPUT_PATH 包含完整历史原型库
set(GE_ES_EXTRACT_HISTORY  ON)
set(GE_ES_RELEASE_VERSION  "8.0.RC2")     # 当前新版本号，用于归档步骤
set(GE_ES_RELEASE_DATE     "2026-02-28")  # 可选
set(GE_ES_BRANCH_NAME      "develop") # 可选；若设为 master 则不会归档，仅走代码生成

add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_math
    OPP_PROTO_TARGET  opgraph_math
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
)
```

## 命名规则

### 产物命名

| 产物类型 | 命名规则 | 示例 (ES_LINKABLE_AND_ALL_TARGET=es_math) |
|---------|---------|--------------------------------|
| 动态库 | `lib<ES_LINKABLE_AND_ALL_TARGET>.so` | `libes_math.so` |
| 静态库 | `lib<ES_LINKABLE_AND_ALL_TARGET>.a` | `libes_math.a` |
| Python 包 | `<ES_LINKABLE_AND_ALL_TARGET>-1.0.0-py3-none-any.whl` | `es_math-1.0.0-py3-none-any.whl` |
| 聚合头文件 | `es_<name>_ops.h` | `es_math_ops.h` |


### Target 命名

| Target 类型 | 命名规则 | 示例 (ES_LINKABLE_AND_ALL_TARGET=es_math) |
|------------|---------|--------------------------------|
| 对外库 | `<ES_LINKABLE_AND_ALL_TARGET>` | `es_math` |

## 注意事项

1. **ES_LINKABLE_AND_ALL_TARGET 命名**：
   - 建议使用 `es_` 前缀（如 `es_math`, `es_nn`）
   - 使用小写字母和下划线
   - 避免使用特殊字符和 C++ 关键字

2. **历史原型库相关变量**：
   - 历史原型库路径由函数**自动推导**（`${CANN_INSTALL_PATH}/cann/opp/history_registry/<module>`），路径存在且非空时自动启用，调用方无需传参
   - **master 分支**：当 `GE_ES_BRANCH_NAME` 为 `master` 时，会忽略 `GE_ES_EXTRACT_HISTORY`/`GE_ES_RELEASE_VERSION`/`GE_ES_RELEASE_DATE`，仅走纯代码生成模式（历史原型库路径仍会传递，用于生成带重载的 C++ API）
   - `GE_ES_EXTRACT_HISTORY=ON` 且 ops 包中存在历史原型库（自动检测到）即为**完整商发模式**，函数内部自动执行两次 gen_esb（代码生成 + 历史原型库归档&合并），调用方无需任何额外处理（参见示例 9）
   - **版本去重**：若历史原型库的 `index.json` 中已存在与 `GE_ES_RELEASE_VERSION` 相同的版本号，则跳过归档步骤，仅执行代码生成（避免重复归档）；同时将 CANN 安装路径中的历史原型库完整复制到 `OUTPUT_PATH`，方便后续换路径安装
   - 纯历史归档模式（仅设置 `GE_ES_EXTRACT_HISTORY=ON`，ops 包中无历史原型库）：gen_esb 输出到 `OUTPUT_PATH`，只生成 JSON，不生成 C++ API
   - 完整商发模式下，代码生成步骤 gen_esb 不传 `--release_version`，以当前日期为锚点自动选取窗口内历史版本对比；归档步骤使用 `GE_ES_RELEASE_VERSION` 作为新版本号写入历史库
   - 完整商发模式下，归档步骤将已有历史库从 `_AUTO_HISTORY_REGISTRY`（CANN 安装路径，只读）复制到 `OUTPUT_PATH`（构建目录，可写），gen_esb 在 `OUTPUT_PATH` 追加当前版本条目；首次构建（CANN 路径中无已有历史库）时直接输出至 `OUTPUT_PATH`；最终 `OUTPUT_PATH` 包含所有历史版本 + 当前版本的完整合并结果
   - `GE_ES_RELEASE_VERSION` 归档时必须设置，否则归档版本无法识别
   - 历史原型库数据（JSON 文件）通常随 ops 包安装，默认位于 `${CANN_INSTALL_PATH}/cann/opp/history_registry/<package_name>/`


## 依赖要求

- **CMake 版本**: >= 3.16
- **CANN run 包**: 已安装并设置环境变量（或手动指定路径）
- **gen_esb**: 此为内部es api的代码生成二进制工具，已在run包中集成，可以通过`gen_esb` --help信息查看使用说明
- **OPP_PROTO_TARGET**: 必须存在
- **Python3**: >= 3.7（仅 `add_es_library_and_whl` 需要）
- **Python 包**: setuptools, wheel（仅 `add_es_library_and_whl` 需要）,推荐使用setuptools==59.6.0、wheel==0.37.
  1的配套版本或更高版本

## 常见问题

### Q：多个 TARGET 如何共享输出目录？

A: 所有TARGET可以使用同一个 `OUTPUT_PATH`，头文件会按包名组织在子目录中：

```cmake
add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_math
    OPP_PROTO_TARGET  opgraph_math
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output  # 共享路径
)

add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_nn
    OPP_PROTO_TARGET  opgraph_nn
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output  # 共享路径
)

# 输出结构：
# output/
# ├── include/
# │   ├── es_math/
# │   └── es_nn/
# ├── lib64/
# │   ├── libes_math.so, libes_math.a
# │   └── libes_nn.so, libes_nn.a
# └── whl/
#     ├── es_math-1.0.0-py3-none-any.whl
#     └── es_nn-1.0.0-py3-none-any.whl
```

### Q：gen_esb 找不到怎么办？

A: 函数会自动查找 gen_esb，如果失败请检查：

1. 确认已安装 CANN run 包
2. 执行 `source ${ASCEND_HOME_PATH}/set_env.sh`或者设置了正确的CMAKE_MODULE_PATH
3. gen_esb 会从 PATH 环境变量或者从 cmake 文件路径自动推导

函数会输出详细的检测信息，根据日志排查问题。

## 完整示例

```cmake
# ===== 基础设置 =====
cmake_minimum_required(VERSION 3.16)
project(my_es_project LANGUAGES CXX)

# ===== 引入函数（推荐：使用 find_package） =====
list(APPEND CMAKE_MODULE_PATH "${ASCEND_HOME_PATH}/include/ge/cmake")
find_package(GenerateEsPackage REQUIRED)

# ===== 第一部分: `前置要求`：定义原型库 =====
add_library(opgraph_math SHARED
)

# ===== 第二部分: 生成 ES API 包 =====
add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_math
    OPP_PROTO_TARGET  opgraph_math
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
)

# ===== 第三部分: 在应用中使用 =====
add_executable(my_app
    src/main.cpp
    src/inference.cpp
)

# 链接ES_LINKABLE_AND_ALL_TARGET
target_link_libraries(my_app PRIVATE
    es_math  # 自动获得依赖、头文件和库
)
```

**构建命令**：

```bash
# 只需一条命令！
make my_app
```