# 样例使用指导

## 1、功能描述
本样例使用Matmul算子普通属性进行构图，旨在帮助构图开发者快速理解普通属性的定义和使用
## 2、目录结构
```angular2html
python/
├── src/
|   └── make_matmul_graph.py   // sample文件
├── CMakeLists.txt             // 编译脚本
├── README.md                  // README文件
├── run_sample.sh              // 执行脚本
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
可执行程序执行过程中，如果需要DUMP图来辅助定位图编译流程，可以在bash run_sample.sh -t sample_and_run_python 之前设置如下环境变量来DUMP图到执行路径下
```bash
export DUMP_GE_GRAPH=2 
```

## 4、核心概念介绍

### 4.1、构图步骤如下：
- 创建图构建器(用于提供构图所需的上下文、工作空间及构建相关方法)
- 添加起始节点(起始节点指无输入依赖的节点，通常包括图的输入(如 Data 节点)和权重常量(如 Const 节点))
- 添加中间节点(中间节点为具有输入依赖的计算节点，通常由用户构图逻辑生成，并通过已有节点作为输入连接)
- 设置图输出(明确图的输出节点，作为计算结果的终点)

### 4.2、普通属性（Normal Attributes）
**概念说明：**
普通属性是算子的配置参数，用于控制算子的行为；属性值在构图时确定，不会在运行时改变。

**构图 API 特点：**
- 属性值必须在构图时确定，不能是运行时变量
- 如果使用默认值，可以省略对应参数（使用默认值）或显式传入
- ES API 会根据属性类型进行类型检查，确保传入的值类型正确

例如 MatMul 算子原型如下所示，ES 构图生成的API是`MatMul()`，支持在 Python 层使用
```bash
  REG_OP(MatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .OP_END_FACTORY_REG(MatMul)
```
其对应的函数原型为：
- 函数名：MatMul
- 参数：共 5 个，依次为 x1， x2， bias， transpose_x1， transpose_x2
- 返回值：输出 y

**Python API中：**
```
Matmul(x1: Union[TensorHolder, TensorLike], x2: Union[TensorHolder, TensorLike], bias: Optional[Union[TensorHolder, TensorLike]] = None, *, transpose_x1: bool = False, transpose_x2: bool = False) -> TensorHolder:
```
注：
1、使用TensorLike类型表达输入，以支持实参可以直接传递数值的情况