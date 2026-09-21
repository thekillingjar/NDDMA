# Python样例使用指导
## 目录结构

```
├── README.md  样例使用指导
├── config
│   ├── model_generator.py  生成用例所需的模型的脚本
│   ├── add_func.json  用例中udf配置文件
│   ├── add_graph.json  用例中计算图编译配置文件
│   ├── data_flow_deploy_info.json  sample_udf_python用例中部署位置配置文件
│   ├── multi_model_deploy.json  sample_multiple_model用例中部署位置配置文件
│   ├── sample_npu_model_deploy_info.json  sample_npu_model用例中部署位置配置文件
│   ├── sample_pytorch_deploy_info.json  sample_pytorch用例中部署位置配置文件
│   └── invoke_func.json  用例中udf_call_nn的编译配置文件
├── sample_base.py  该样例展示了基本的DataFlow API构图，包含UDF，GraphPp和UDF执行NN推理三种类型节点的构造和执行
├── sample_udf_python.py  该样例展示了python dataflow调用 udf python的过程
├── sample_exception.py  该样例展示了使能异常上报的样例
├── sample_pytorch.py  该样例展示了DataFlow结合pytorch进行模型的在线推理
├── sample_npu_model.py  该样例展示了DataFlow在线推理时，udf使用分别npu_model实现模型下沉和数据下沉场景
├── sample_multiple_model.py  该样例将pytorch模型直接通过装饰器构造成funcPp，同时和执行onnx模型/pb模型的GraphPp进行串接
├── sample_perf.py  该样例为性能打点样例
├── udf_py
│   ├── udf_add.py  使用python实现udf多func功能
│   └── udf_control.py  使用python实现udf功能，用于控制udf_add中多func实际执行的func
└── udf_py_ws_sample  完整样例用于说明python udf实现
    ├── CMakeLists.txt  udf python完整工程cmake文件样例
    ├── func_add.json   udf python完整工程配置文件样例
    ├── src_cpp
    │   └── func_add.cpp  udf python完整工程C++源码文件样例
    └── src_python
        └── func_add.py  udf python完整工程python源码文件样例

```

## 环境准备
- 参考[环境准备](../../../README.md#环境准备)下载安装驱动/固件/CANN软件包；

- python 版本要求：python3.11 具体版本以dataflow wheel包编译时用的python版本为准，如果需要使用不同python版本，可以参考[编译](../../../docs/build.md#编译)重新编译ge_compiler包并安装；

- config目录下的模型生成脚本 model_generator.py 依赖tensorflow-cpu和onnx，需通过 pip3 install tensorflow-cpu 和 pip3 install onnx 安装；

- [sample_pytorch.py](sample_pytorch.py)、[sample_npu_model.py](sample_npu_model.py) 样例依赖torch_npu和torchvision包，torch_npu需要根据实际环境，安装对应的**torch**与**torch_npu**包(建议使用大于等于2.1.0的版本， [获取方法](https://gitcode.com/Ascend/pytorch))，torchvision与torch有配套关系，需要等torch安装完之后使用pip3 install torchvision安装。

## 运行样例
下文中numa_config.json文件配置参考[numa_config字段说明及样例](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/graph/dataflowcdevg/dfdevc_23_0031.html)
```bash
# 执行config目录下的tensorflow原始模型生成脚本：
python3 config/model_generator.py
# 执行结束后，在config目录下生成add.pb模型和simple_model.onnx模型

# 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       #0 debug 1 info 2 warn 3 error 不设置默认error级别
# 必选
source {HOME}/Ascend/cann/set_env.sh #{HOME}为CANN软件包安装目录，请根据实际安装路径进行替换
export RESOURCE_CONFIG_PATH=xxx/xxx/xxx/numa_config.json

python3.11 sample_base.py
python3.11 sample_udf_python.py
python3.11 sample_exception.py
python3.11 sample_pytorch.py
python3.11 sample_npu_model.py
python3.11 sample_multiple_model.py
python3.11 sample_perf.py

# unset此环境变量，防止影响非dflow用例
unset RESOURCE_CONFIG_PATH

```

