## 简介
因为ES构图API是基于IR原型生成, 部分算子IR无法支撑所有信息表达(Conv)或者IR表达有冗余信息(IdentityN), 所以推荐用户自行封装ES API, 同时支持用户自定义行为。<br>
该Sample旨在指导用户如何自定义ES API <br>
**注:** Conv2D等算子在默认生成API中, 直接使用会有Format问题, 需要自行添加逻辑并生成自定义ES API(见 [Sample 1](#sample1))
## 快速开始
1. 通过[安装指导](../../docs/build.md#2-安装软件包)正确安装`toolkit`和`ops`包, 并正确配置环境变量
2. 通过命令`bash run_sample.sh` 运行脚本 [run_sample.sh](run_sample.sh)

## 预期结果
项目build目录下生成pbtxt格式dump图

## 实现步骤
### 步骤一. [可选] 生成ES API代码参考
通过gen_esb可执行程序生成ES_API代码 (参考 [gen_esb_readme.md](../../docs/es/tools/gen_esb.md))
### 步骤二. 自定义ES API代码
参考步骤一生成的代码编写自定义API代码, 以下为代码样例
- Sample 1. 合法性补全 Conv2D <a id="sample1"></a> <br>
  Conv2D类型算子的Format需要自行设置, 可以增加逻辑保证算子有效
  - [my_Conv2D_c.h](./custom/my_Conv2D_c.h)
  - [my_Conv2D.h](./custom/my_Conv2D.h)
  - [my_Conv2D.cpp](./custom/my_Conv2D.cpp)

- Sample 2. 冗余属性消除 ConcatD <br>
  由于历史原因，部分 IR 中定义的属性存在冗余 <br>
  例如在 ConcatD 算子中，属性 N 表示输入 x 的数量 <br>
  优化方式是将属性 N 注册为可推导属性，使其由 x_num 自动推导。通过如下语句注册推导逻辑：`int64_t N = x_num;`
  - [my_ConcatD_c.h](./custom/my_ConcatD_c.h)
  - [my_ConcatD.h](./custom/my_ConcatD.h)
  - [my_ConcatD.cpp](./custom/my_ConcatD.cpp)

- Sample 3. 动态输出参数消除 IdentityN <br>
  在生成 API 时将内嵌算子注册动态输出数量的推导逻辑，从而自动确定输出数量，无需显式传参 <br>
  例如IdentityN可以新增推导代码，表达动态输出 y 的数量可以由"x_num"表达式获得: `int64_t y_num = x_num;`
  - [my_IdentityN_c.h](./custom/my_IdentityN_c.h)
  - [my_IdentityN.h](./custom/my_IdentityN.h)
  - [my_IdentityN.cpp](./custom/my_IdentityN.cpp)

- Sample 4. 传递自定义属性
  - [my_Conv2D_c.h](./custom/my_Conv2D_c.h)
  - [my_Conv2D.h](./custom/my_Conv2D.h)
  - [my_Conv2D.cpp](./custom/my_Conv2D.cpp)


### 步骤三. 构图
自定义构图代码逻辑 (参照 [main.cc](./src/main.cc))
### 步骤四. 构建工程
关键流程如下 (参照 [CMakeLists.txt](CMakeLists.txt))
1. 设置变量 && 引入cmake函数
2. 定义算子原型库
    ```cmake
    # 注意: 
    # 当前为开箱即用示例, 即定义INTERFACE使用已有的原型so
    # 正常使用不会是用现有的so，更普遍的方式是直接使用项目中已有的so的target
    # 使用项目中已有的so方式:
    # 1. add_library(opgraph_all SHARED)
    # 2. 设置PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    add_library(opgraph_all INTERFACE)
    set_target_properties(opgraph_all PROPERTIES
        INTERFACE_LIBRARY_OUTPUT_DIRECTORY "${ASCEND_INSTALL_PATH}/opp/built-in/op_proto"
    )
    ```
3. 通过`add_es_library_and_whl` 或者 `add_es_library` 生成ES API包, 如果需要屏蔽部分算子ES API生成, 则加入参数EXCLUDE_OPS
    ```cmake
    add_es_library(
        ES_LINKABLE_AND_ALL_TARGET es_all
        OPP_PROTO_TARGET  opgraph_all
        OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
        EXCLUDE_OPS       Conv2D,ConcatD
    )
    ```

4. 引入构建自定义ES API包
    ```cmake
    file(GLOB CPP_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/custom/*.cpp")
    add_library(self_defined_es_lib SHARED ${CPP_FILES})
    
    find_library(GRAPH_LIB NAMES graph)
    
    # 添加头文件路径
    target_include_directories(self_defined_es_lib
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/custom
    )
    ```
5. 自定义 API包 link
    ```cmake
    target_link_libraries(self_defined_es_lib PUBLIC es_all)
    
    add_dependencies(self_defined_es_lib es_all)
    ``` 

## 注意事项
1. 确保环境变量已正确设置
2. 确保有足够的磁盘空间存储生成的代码文件
3. 生成的代码文件数量取决于系统中注册的算子数量