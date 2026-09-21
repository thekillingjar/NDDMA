## 目录结构
```
├── 01_udf_add   
│   ├── CMakeLists.txt   
│   └── add_flow_func.cpp 单fun接口定义的add函数   
├── 02_udf_call_add_nn   
│   ├── CMakeLists.txt   
│   └── call_nn_flow_func.cpp UDF调用tensorflow模型  
├── 03_udf_add_multi_func   
│   ├── CMakeLists.txt   
│   └── add_flow_func.cpp 多Func接口定义add函数  
├── 04_control_func   
│   ├── CMakeLists.txt   
│   └── control_func.cpp 根据输入激活03中定义的多func中的某一个  
└── README.md
```

## 开发指导
01_udf_add 和 02_udf_call_add_nn 为单func调用样例。
1. 包含对外头文件 meta_flow_func.h
2. 定义处理类，public继承MetaFlowFunc基类，重载Init和Proc两个函数
* Init函数执行初始化动作，如变量初始化，获取属性等。UDF框架在初始化阶段会调用该函数
* Proc中是用户自定义的计算处理逻辑，UDF框架在接收到输入数据后，会调用该函数
3. UDF注册，通过REGISTER_FLOW_FUNC宏实现将类声明映射成用户自定义的UDF函数名，同时注册到UDF框架中。如：    
   REGISTER_FLOW_FUNC("call_nn", CallNnFlowFunc);    
   call_nn：用户自定义的UDF函数名       
   CallNnFlowFunc：实际执行的类名，需要和cpp文件中的类名保持一致

03_udf_add_multi_func和04_control_func 为多func调用样例
1. 包含对外头文件 meta_multi_func.h
2. 定义处理类，public继承MetaMultiFunc基类，重载Init和Proc两个函数
* Init函数执行初始化动作，如变量初始化，获取属性等。UDF框架在初始化阶段会调用该函数
* Proc中是用户自定义的计算处理逻辑，UDF框架在接收到输入数据后，会调用该函数，proc函数可以定义多个
3. UDF注册，通过FLOW_FUNC_REGISTRAR宏实现将类声明映射成用户自定义的UDF函数名，同时注册到UDF框架中。如：    
   FLOW_FUNC_REGISTRAR(AddFlowFunc)    
   .RegProcFunc("Proc1", &AddFlowFunc::Proc1)    
   .RegProcFunc("Proc2", &AddFlowFunc::Proc2);
## 编译指导
UDF函数开发完成后，可以使用以下编译指令查看CMakeLists文件及cpp源码是否存在问题。
```bash
source {HOME}/Ascend/cann/set_env.sh #{HOME}为CANN软件包安装目录，请根据实际安装路径进行替换
# 以01_udf_add为例
cd 01_udf_add
mkdir build
cd build
# 如果host是x86 编译x86类型的so
cmake -S .. -DTOOLCHAIN=g++ -DRELEASE=../release -DRESOURCE_TYPE=x86 -DUDF_TARGET_LIB=udf
# 如果host是arm 编译arm类型的so
cmake -S .. -DTOOLCHAIN=g++ -DRELEASE=../release -DRESOURCE_TYPE=Aarch -DUDF_TARGET_LIB=udf
# 如果编译Ascend类型的so，需要指定安装了cann包的环境中的g++
cmake -S .. -DTOOLCHAIN=$ASCEND_HOME_PATH/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ -DRELEASE_DIR=../release -DRESOURCE_TYPE=Ascend -DUDF_TARGET_LIB=udf
make -j 64
```