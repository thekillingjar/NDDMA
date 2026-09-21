# 样例使用指导

## 1、功能描述
本样例使用动态输出算子进行构图，旨在帮助构图开发者快速理解动态输出的定义和使用
## 2、目录结构
```angular2html
cpp/
├── src/
|   └── CMakeLists.txt           // CMake构建文件
|   └── es_showcase.h            // 头文件
|   └── make_split_graph.cpp     // sample文件
├── CMakeLists.txt               // CMake构建文件
├── main.cpp                     // 程序主入口
├── README.md                    // README文件
├── run_sample.sh                // 执行脚本
├── utils.h                      // 工具文件
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
可通过data文件查看计算结果
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

### 4.2、动态输出
**概念说明：**
动态输出是指某些算子的输出的个数不固定；例如 Split 算子，该算子为动态多输出算子

**构图 API 特点：**
- 需要一个参数来传递动态输出的个数

例如 Split 算子原型如下所示，ES 构图生成的API是 Split（C++）或 EsSplit（C）
```bash
  REG_OP(Split) 
    .INPUT(split_dim, TensorType({DT_INT32})) 
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, 
     DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, 
     DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, 
     DT_BF16, DT_BOOL})) 
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, 
     DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, 
     DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, 
     DT_BF16, DT_BOOL})) 
    .REQUIRED_ATTR(num_split, Int) 
    .OP_END_FACTORY_REG(Split)
```
其对应的函数原型为：
- 函数名：Split（C++）或 EsSplit（C）
- 参数：共 4 个，依次为 split_dim， x， y_num， num_split
- 返回值：输出 y

注：因为当前 IR 定义中未显式描述“输入/属性 → 动态输出数量”的映射关系，导致构图 API 难以自动推导输出个数，因此需要手动指定输出数量；
若需自定义API，请参考 [自定义ES_API.md](../../../custom_es_api/README.md)

**C API中：**
```
typedef struct {
  EsCTensorHolder **y;
  int64_t y_num;
} EsSplitOutput;
EsSplitOutput* EsSplit(EsCTensorHolder *split_dim, EsCTensorHolder *x, int64_t y_num, int64_t num_split);
```
**C++ API：**
```
std::vector<EsTensorHolder> Split(const EsTensorLike &split_dim, const EsTensorLike &x, int64_t y_num, int64_t num_split);
```
注：使用TensorLike类型表达输入，以支持实参可以直接传递数值的情况