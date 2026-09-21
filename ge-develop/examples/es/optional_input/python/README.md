# 样例使用指导

## 1、功能描述
本样例使用BatchNorm算子可选输入进行构图，旨在帮助构图开发者快速理解可选输入定义和使用该类型算子进行构图
## 2、目录结构
```angular2html
python/
├── src/
|   └── make_batchnorm_graph.py   // sample文件
├── run_sample.sh                 // 执行脚本
├── CMakeLists.txt                // 编译脚本
├──README.md                     // README文件
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

### 4.2、概念说明：
可选输入是指算子的某些输入是非必选输入。

**构图 API 特点：**
- 构图时输入为非必传参数

例如 BatchNorm 算子原型如下所示，ES 构图生成的API是`BatchNorm()`，支持在 Python 层使用
```bash
  REG_OP(BatchNorm)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .ATTR(exponential_avg_factor, Float, 1.0)
    .OP_END_FACTORY_REG(BatchNorm)
```
其对应的函数原型为：
- 函数名：BatchNorm
- 参数：共 9 个，依次为 x， scale， offset， mean(可选输入)， variance(可选输入)， epsilon， data_format， is_training， exponential_avg_factor
- 返回值：输出 y， batch_mean， batch_variance， reserve_space_1， reserve_space_2， reserve_space_3

**Python API中：**
```bash
BatchNorm(x: Union[TensorHolder, TensorLike], scale: Union[TensorHolder, TensorLike], offset: Union[TensorHolder, TensorLike], mean: Optional[Union[TensorHolder, TensorLike]] = None, variance: Optional[Union[TensorHolder, TensorLike]] = None,
epsilon: float = 0.00100, data_format: str = "NHWC", is_training: bool = True, exponential_avg_factor: float = 0.00100) -> BatchNormOutput:
```

```bash
class BatchNormOutput:
    def __init__(self, y: TensorHolder, batch_mean: TensorHolder, batch_variance: TensorHolder, reserve_space_1: TensorHolder, reserve_space_2: TensorHolder, reserve_space_3: TensorHolder)
        self.y = y
        self.batch_mean = batch_mean
        self.batch_variance = batch_variance
        self.reserve_space_1 = reserve_space_1
        self.reserve_space_2 = reserve_space_2
        self.reserve_space_3 = reserve_space_3
```
注： 
1.使用TensorLike类型表达输入，以支持实参可以直接传递数值的情况

## Python 层 API 示例

### 方式 : 直接调用 BatchNorm()

```python
from ge.es.graph_builder import GraphBuilder, TensorHolder
from ge.graph import Tensor
from ge.graph.types import DataType, Format
from ge.graph import Graph
from ge.es.all import BatchNorm

# 1. 创建图构建器
builder = GraphBuilder("control_dep_example")
# 2. 创建节点
input_tensor_holder = builder.create_input(
    index=0,
    name="input",
    data_type=DataType.DT_FLOAT,
    shape=[2, 3]
)
variance = builder.create_input(
    index=1,
    name="variance",
    data_type=DataType.DT_FLOAT,
    shape=[2, 3]
)
scale = builder.create_vector_int64([3, 1])
offset = builder.create_vector_int64([3, 0])
# 3. 可选输入 mean 为None, variance 有输入
batchNorm_tensor_holder = BatchNorm(input_tensor_holder, scale, offset, None, variance)
# 4. 设置输出并构建
builder.set_graph_output(batchNorm_tensor_holder.y, 0)
graph = builder.build_and_reset()
```