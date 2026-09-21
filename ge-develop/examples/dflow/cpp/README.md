# C++样例使用指导
## 目录结构

```
├── CMakeLists.txt  cmake配置文件
├── README.md  样例使用指导
├── config    
│   ├── model_generator.py  生成用例所需的模型的脚本
│   ├── add_func_multi.json  用例中使用的多func的配置文件   
│   ├── add_func_multi_control.json  用例中使用的多func的配置文件    
│   ├── add_func.json  用例中add FunctionPp对应的配置文件  
│   ├── add_graph.json  用例中add GraphPp对应的配置文件  
│   ├── data_flow_deploy_info.json  用例中指定节点部署位置的配置文件
│   └── invoke_func.json  用例中udf调用nn对应的配置文件  
├── node_builder.h  构造functionPp和GraphPp的公共方法  
├── sample_base.cpp  该样例展示了基本的DataFlow API构图，包含UDF，GraphPp和UDF执行NN推理三种类型节点的构造和执行
├── sample_timebatch.cpp  该样例展示了TimeBatch使用方法
├── sample_countbatch.cpp  该样例展示了CountBatch使用方法
├── sample_tensorflow.cpp  该样例展示了Tensorflow图构造Dataflow节点的方法
├── sample_multifunc.cpp  该样例展示了多func的调用方法
├── sample_exception.cpp  该样例展示了开启异常上报的方法 
└── sample_perf.cpp  该样例测试Feed和Fetch的接口性能
```


## 环境要求
- 参考[环境准备](../../../README.md#环境准备)下载安装驱动/固件/CANN软件包；
- config目录下的模型生成脚本 model_generator.py 依赖tensorflow，需通过pip3 install tensorflow安装。

## 程序编译
```bash
# 执行config目录下的tensorflow原始模型生成脚本：
python3 config/model_generator.py
# 执行结束后，在config目录下生成add.pb模型

source {HOME}/Ascend/cann/set_env.sh  # "{HOME}/Ascend"为CANN软件包安装目录，请根据实际安装路径进行替换。
mkdir build
cd build
cmake .. && make -j 64
cd ../output
```

## 运行样例
下文中numa_config.json文件配置参考[numa_config字段说明及样例](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/graph/dataflowcdevg/dfdevc_23_0031.html)
```bash
# 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       #0 debug 1 info 2 warn 3 error 不设置默认error级别
# 必选
source {HOME}/Ascend/cann/set_env.sh # "{HOME}/Ascend"为CANN软件包安装目录，请根据实际安装路径进行替换。
export RESOURCE_CONFIG_PATH=xxx/xxx/xxx/numa_config.json

./sample_base
./sample_timebatch
./sample_countbatch
./sample_tensorflow
./sample_multifunc
./sample_exception
./sample_perf

# unset此环境变量，防止影响非dflow用例
unset RESOURCE_CONFIG_PATH

```
