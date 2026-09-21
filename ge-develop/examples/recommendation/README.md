# 样例使用指导

## 功能描述

本样例是一个推荐网络的高性能推理示例，演示了如何利用**GE API**和**ACL API**实现推理流程，并通过**多实例**、**批量H2D(Host-to-Device)内存拷贝** 和**AI Core控核**技术来优化推荐网络的推理吞吐性能。

## 目录结构

```
├── src
│   ├──model_inference.cpp               // 模型推理实现文件
│   ├──model_inference.h                 // 模型推理头文件
│   ├──recomand_random_input.cpp         // 推理客户端，构造随机数据，调用模型推理
├── CMakeLists.txt                       // 编译脚本
```

## 环境要求
- 已完成[昇腾AI软件栈在开发环境上的部署](../../docs/build.md#2-安装软件包)

## 实现步骤
1. 图构建：使用aclgrphParseTensorFlow解析模型文件，构建GE计算图。
2. 图编译与加载：通过GE API(ge::Graph, ge::Session)进行图的编译(Compile)和加载(Load)。
3. 数据准备与执行：根据模型输入结构构造随机数据，使用GE API进行推理。
4. 性能优化： 
   - 多实例：通过多线程创建多个推理实例，提升系统并发处理能力。
   - 控核：在创建ge::Session时，通过options参数指定单算子可使用的AI Core数量。
   - 批量H2D：使用aclrtMemcpyBatch接口合并多次内存拷贝操作，减少开销。

## 构建验证

假设toolkit的安装目录为install_path, 例如`/home/HwHiAiUser/Ascend/cann/`

1. 配置环境变量。
   ```bash
   source ${install_path}/set_env.sh
   ```
2. 执行如下命令，创建data目录，并[下载](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ge/DCN_v2.pb)模型pb文件，放入data目录。
   ```shell
   mkdir data
   ```
3. 执行如下命令，编译生成可执行文件
   ```
   mkdir build && cd build
   cmake ..
   make
   ```
   执行后，在**build**目录下产生recomand_exec可执行文件

4. 执行如下命令，测试推荐网络不开优化特性时的推理吞吐性能。
   ```shell
   ./recomand_exec
   ```
5. 测试开启4个多实例、开启批量H2D、控核时的网络推理性能，其中aiCoreNum参考[GE图引擎接口 -> 数据类型 -> options参数说明](https://www.hiascend.com/document/redirect/CannCommunityAscendGraphApi)按照实际硬件信息调整。
   ```shell
   ./recomand_exec --multiInstanceNum=4 --enableBatchH2D=true --aiCoreNum="16|16"
   ```
## 性能差异对比
在Ascend 910C平台测试不同配置下的吞吐量（TPS）与时延（ms）表现：

 | 配置方案                    | 吞吐 / 时延               |
 | --------------------------- |-----------------------|
 | 单实例                      | 745,55 TPS / 1.471ms  |
 | 单实例 + 批量H2D             | 131,191 TPS / 0.792ms |
 | 多实例（4）                  | 155,104 TPS / 2.089ms |
 | 多实例（4）+ 控核（16\|16）    | 185,415 TPS / 1.797ms |
 | 多实例（4）+ 控核（16\|16）+ 批量H2D | 251,877 TPS / 1.285ms |