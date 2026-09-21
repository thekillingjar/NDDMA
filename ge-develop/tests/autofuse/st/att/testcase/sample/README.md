# ATT(Ascend Tiling Tools) 端到端流程Sample
## 一.介绍
ATT提供一套端到端流程的验证工程，用户可以将工程部署到自己指定的目录，并基于该工程进行算子ATT自动生成的开发/验证，以下是工程的目录及使用流程介绍。
## 二.目录介绍
```angular2html
├── depends                 # 依赖的其他文件
├── examples                # 样例文件
├── gen_data_st             # 用于验证TilingFunc生成数据的ST工程，可以在gen code后由用户修改
├── gen_code_st             # 用于生成TilingFunc的ST工程
├── user_inc                # 用户需要修改的头文件
├── user_src                # 用户需要修改的构图代码
```
## 三.前置依赖
### 3.1 环境准备
在源码编译前，请确保环境满足如下要求：

- 已安装CANN开发套件包（Ascend-cann-toolkit_\<cann_version>\_linux\_\<arch>）。

  CANN开发套件软件包请从[Link](https://www.hiascend.com/developer/download/community/result?module=cann)获取，支持的安装方式及操作系统请参见配套版本的[用户手册](https://hiascend.com/document/redirect/CannCommunityInstSoftware)。

  **CANN开发套件包安装命令示例：**

    - 使用默认路径安装
      ```shell
      ./Ascend-cann-toolkit_<cann_version>_linux_<arch>.run --install
      ```

      - 若使用root用户安装，安装完成后相关软件存储在`/usr/local/Ascend/ascend-toolkit/latest`路径下。
      - 若使用非root用户安装，安装完成后相关软件存储在`$HOME/Ascend/ascend-toolkit/latest`路径下。

    - 指定路径安装

      ```bash
      # CANN开发套件包安装命令示例：
      ./Ascend-cann-toolkit_<cann_version>_linux-<arch>.run --install --install-path=${install_path}
      ```

      安装完成后，相关软件存储在`${install_path}`指定路径下。

- 源码编译存在如下依赖，若环境中不存在，请自行安装。
    - gcc：9.5.0版本及以上 (建议9.5.0)
    - cmake：3.20.0版本及以上 (建议3.20.0）
    - compile cache，编译器缓存优化工具，加快二次编译速度
      ```bash
      # 安装命令示例：
      sudo apt-get install ccache
      ```
### 3.2 源码下载
  ```bash
  git clone https://gitee.com:ascend/ascgen-dev.git --recursive
  ```
### 3.3 编译第三方开源库
在软件编译前，需要进行第三方软件库的编译，在ascgen仓根目录下执行调用
```shell
bash build_third_party.sh
```
注意：在开发环境使用时需要配置代理：
```shell
export http_proxy=http://域账号:域密码@proxy.huawei.com:8080
export https_proxy=$http_proxy
export ftp_proxy=$http_proxy
git config --global http.sslVerify false
git config --global http.proxy https://域账号:域密码@proxy.huawei.com:8080
git config --global https.proxy https://域账号:域密码@proxy.huawei.com:8080
```
### 3.4 编译ATT库
Sample工程依赖ATT库文件（后续ATT库归档后该步骤可以优化为可选），在ascgen仓根目录下执行：
```shell
cd att/test
bash run_test.sh -s --ascend_install_path=${install_path}
```
会编译出sample所依赖的ATT库(libatt.so)，并执行相应的ST。
**注意：需要通过--ascend_install_path命令指定ASCEND_INSTALL_PATH的目录（设置为3.1 环境准备中安装的路径）。**
## 四.部署ATT端到端工程
前置依赖准备好后，可以将ATT的端到端流程的工程部署到指定的目录${output_path}，在ascgen仓根目录下执行：
```shell
cd att/sample
bash install.sh ${output_path}
```
**注意：代码仓下已有的sample目录部分配置未初始化，需要经部署动作设置为正确的值，该部署动作为必选动作**。
## 五.构建ST工程
完成ATT端到端工程的部署后，即可基于端到端工程进行开发及验证。
### 5.1 自定义Ascend内置算子注册（可选）
ATT本身提供了部分内置的基本算子，当前支持的内置算子有：
#### 输入算子
Data/Constant/Workspace/TbufData；
#### 计算API
Broadcast/Nop/Cast/Abs/Exp/Max/Sum/Add/Sub/Div/Mul/GT/Muls/MatMul/FlashSoftmax/Dropout/Select；
#### 内存操作API
Load/Store；
#### 更多的Ascend内置算子
定义在：att/ascir/generator/v1/ascir_builtin_ops.cpp，若用户需要扩展算子的使用，需要基于att/sample/user_src/custom_ascir.cpp注册自定义算子，并基于att/sample/user_src/custom_api_perf.cpp扩展性能公式。
注意：开发自定义API的性能公式需要引入头文件ascir_custom_ops.h（该头文件会在编译时自动生成）。
### 5.2 自定义Ascend算子性能公式（可选）
ATT本身提供了部分内置的性能公式，当前支持的性能公式有：
#### 内存搬运
T_LoadTscm/CopyL2ToL1/T_LoadA/T_LoadB/CopyL0CToL2/T_FixPipeTrans/Load/Store；
#### 计算API
VectorCompute/T_Mmad/Abs/Sub/Div/Exp/Max/Sum/Cast/Broadcast/Mul/Add/Where/Neg/Rsqrt/Reduction/Store_Reduction/To_Type/Sigmoid/Muls/FlashSoftmax/Dropout/MatMul/Select/Constant；
#### 更多Ascend内置性能公式
定义在att/gen_model_info/api_perf_register/ascendc_api_perf.cpp。
若用户需要自定义AscendC API类型，应同步增加性能公式的注册，需要基于att/sample/user_src/custom_api_perf.cpp注册算子的自定义性能公式扩展算子的性能公式表达。
### 5.3 Ascend构图
ascir(Ascend IR)用于描述AscendC算子内部数据流/调度逻辑(轴的切分/合并/轴的执行顺序等信息)/内存分配（Buffer分配）等信息。当前ascir提供了一套构图接口，支持AscendC算子分别进行数据流/Tiling切分策略/Buffer分配三个阶段的构图。
工程中构图文件位于user_src/custom_ascend_graph.cpp，构图可以参考examples里的示例。
#### 5.3.1 修改Option
若用户希望修改传入的Option，可以在custom_ascend_graph.cpp的GeneratorAttOptions函数中设置(当前的Option列表可以参考gen_tiling_impl.h中的介绍)
#### 5.3.2 多模板构造
需要用户在custom_ascend_graph.cpp中定义多个模板的BuildOriginGraph/AddScheInfoToGraph/AddBuffInfoToGraph对应的函数，并在GenerateAscGraphs函数中添加到输出graphs中(tiling_key可以自行指定)。
### 5.4 构建ATT Tiling工程
用户完成Ascend构图后，就可以对工程进行编译构建，在之前安装的${output_path}/sample目录下执行(注意不是原工程目录，而是install.sh脚本安装后的目录)：
```shell
bash build.sh --build_type=gen --target=code --ascend_install_path=${install_path}
```
**注意1：当前工程提供样例，使用命令:**
```shell
bash build.sh --example=flash_attention --build_type=gen --target=code --ascend_install_path=${install_path}
```
可以测试样例程序并生成样例的工程。
编译构建用于生成tiling_func.cpp/tiling_data.h及验证结果需要的input_shapes.json及exec_tiling_func.cpp到gen_data_st目录内。

**注意2：需要通过--ascend_install_path命令指定ASCEND_INSTALL_PATH的目录（设置为3.1 环境准备中所安装的路径）。**
### 5.5 二次加工TilingFunc（可选）
用户可以按照自己的需要自行修改ATT生成的TilingFunc(gen_data_st目录下的tiling_func.cpp及tiling_data.h)。
### 5.6 修改ST工程输入
为了测试Tiling的结果，需要用户根据需求修改input_shapes.json及exec_tiling_func.cpp以打印自己期望的结果。
#### 5.6.1 如何修改工程输入
在生成的gen_data_st/input_shapes.json中描述了各个轴的Shape信息，用户期望可以修改该文件调整算子输入：
```json
{
    "input_shapes": [
        {
            "axes": [
                [
                    "B",
                    1
                ],
                [
                    "N",
                    1
                ],
                [
                    "G",
                    1
                ],
                [
                    "S1",
                    1
                ],
                [
                    "S2",
                    1
                ],
                [
                    "D",
                    1
                ]
            ],
            "tiling_key": 0
        }
    ]
}
```
#### 5.6.2 如何修改工程输出打印
在生成的gen_data_st/exec_tiling_func.cpp文件中会调用ATT的TilingFunc，用户期望可以修改该文件来打印自己期望的结果：
```c++
if (optiling::GetTiling(tiling_data, input_shape.tiling_key)) {
    GELOGI("Get tiling info successfully, you can print some data what you want to see.");
    // TODO output can be modified by user
    // 用户可以在这里增加打印执行的结果
    GELOGI("Get tiling data:get_block_dim=%u, get_s2t_size=%u, get_s1t_size=%u, get_bngs1Tb_size=%u, "
            "get_s1tt_size=%u, get_s1tt_size=%u, get_s1tt2_size=%u",
            tiling_data.get_block_dim(), tiling_data.get_s2t_size(), tiling_data.get_s1t_size(),
            tiling_data.get_bngs1Tb_size(), tiling_data.get_s1tt_size(), tiling_data.get_s1tt_size(),
            tiling_data.get_s1tt2_size());
    GELOGI("Get tiling data info of tiling key[%u]", input_shape.tiling_key);
  } else {
    GELOGE(ge::FAILED, "Error:failed to get tiling, tiling_key[%u]!", input_shape.tiling_key);
    return ge::FAILED;
  }
}
```

### 5.7 生成ST工程输出
完成ST工程的修改后，在sample目录下执行：
```shell
bash build.sh --build_type=gen --target=data --ascend_install_path=${install_path}
```
对ST工程进行编译和执行，用户可以校验获得的结果是否符合自己的预期。
