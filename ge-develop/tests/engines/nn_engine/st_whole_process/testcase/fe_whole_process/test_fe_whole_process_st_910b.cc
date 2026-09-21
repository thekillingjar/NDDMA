/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <regex>
#include <fstream>
#include <nlohmann/json.hpp>
#include "framework/fe_running_env/include/fe_running_env.h"
#include "register/graph_optimizer/fusion_common/fusion_turbo.h"
#define protected public
#define private public
#include "framework/fe_running_env/include/fe_env_utils.h"
#include "graph_optimizer/fe_graph_optimizer.h"
#include "inc/ffts_configuration.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph_constructor/graph_constructor.h"
#include "graph/manager/graph_var_manager.h"
#include "platform/platform_info.h"
#undef protected
#undef private
#include "stub/te_fusion_api.h"

extern std::string g_runtime_stub_mock;
using namespace std;

namespace fe {
class STestFeWholeProcess910B: public testing::Test
{
 protected:
  static void SetUpTestCase() {
    auto &fe_env = fe_env::FeRunningEnv::Instance();
    std::map<string, string> new_options;
    new_options[ge::PRECISION_MODE] = fe::ALLOW_FP32_TO_FP16;
    new_options[ge::SOC_VERSION] = "Ascend910B1";
    new_options[ge::AICORE_NUM] = "24";
    const char *soc_version = "Ascend910B1";
    rtSetSocVersion(soc_version);
    fe_env.ResetEnv(new_options);
    g_runtime_stub_mock = "rtMemcpyAsync";  // rts MemcpyAsync 会生成失败, 这里打桩rts函数
  }

  static void TearDownTestCase() {
    te::TeStub::GetInstance().SetImpl(std::make_shared<te::TeStubApi>());
    auto &fe_env = fe_env::FeRunningEnv::Instance();
    fe_env.FinalizeEnv();
    g_runtime_stub_mock = "";
  }
};

//TEST_F(STestFeWholeProcess910B, test_fixpipe_ub_fusion) {
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//  std::map<string, string> session_options;
//  session_options[ge::PRECISION_MODE] = fe::ALLOW_FP32_TO_FP16;
//  session_options[ge::JIT_COMPILE] = "1";
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("gpt2.txt");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//  EXPECT_EQ(graph->GetName(), "gpt2");
//  EXPECT_EQ(graph->GetDirectNode().size(), 841);
//  /* 在执行前设置需要dump图的阶段 */
//  DUMP_GRAPH_WHEN("AfterCalcOpParam");
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//  CHECK_GRAPH(AfterCalcOpParam) {
//    for (auto &node : graph->GetDirectNode()) {
//      if (node->GetName() == "megatron/gpt2_block_0_recompute/TransformerLayer/MLP/Gelu_copy") {
//        auto input_nodes = node->GetInDataNodes();
//        auto output_nodes = node->GetOutDataNodes();
//        ASSERT_EQ(output_nodes.size(), 2);
//        int count_check = 0;
//        for (const auto &output : output_nodes) {
//          if (output->GetType() == fe::MATMULV2OP) {
//            count_check++;
//          }
//
//        if (output->GetType() == "GeluGrad") {
//          EXPECT_EQ(output->GetName(),
//                    "loss_scale/Compute_gard/gradients/megatron/gpt2_block_0_recompute/TransformerLayer/MLP/Gelu_grad/GeluGrad");
//          count_check++;
//          }
//        }
//        EXPECT_EQ(count_check, 1);
//      }
//    }
//  };
//}

// TEST_F(STestFeWholeProcess910B, test_ffts_context) {
//   te::TeStub::GetInstance().SetImpl(std::make_shared<te::RealTimeCompilation>());
//   auto &fe_env = fe_env::FeRunningEnv::Instance();
//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//   std::map<string, string> session_options;
//   session_options[ge::PRECISION_MODE] = fe::ALLOW_FP32_TO_FP16;
//   session_options[ge::JIT_COMPILE] = "1";
//   string network_path = fe_env::FeRunningEnv::GetNetworkPath("resnet18.txt");
//   bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//   ASSERT_EQ(state, true);
//   EXPECT_EQ(graph->GetName(), "resnet18");
//   EXPECT_EQ(graph->GetDirectNode().size(), 870);

//   //set runmode to ffts
//   ffts::Configuration::Instance().SetFftsOptimizeEnable();
//   /* 在执行前设置需要dump图的阶段 */
//   DUMP_GRAPH_WHEN("MergedComputeGraphAfterCompositeEnginePartition", "AfterGetTask");
//   EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//   CHECK_GRAPH(MergedComputeGraphAfterCompositeEnginePartition) {
//     // 校验切分的算子数
//     int32_t count = 0;
//     std::regex reg("FusedBatchNormV3.*_lxslice[0-9]+$");
//     for (auto &node : graph->GetAllNodes()) {
//       std::cmatch base_match;
//       if (std::regex_search(node->GetName().c_str(), base_match, reg)) {
//         ++count;
//       }
//     }
//     // EXPECT_EQ(count, 202);
//   };

//   CHECK_GRAPH(AfterGetTask) {
//     // 校验atomic_clean_ptr
//     int32_t count = 0;
//     for (auto &node : graph->GetAllNodes()) {
//       if (node->GetName().find("FusedBatchNormV3") != string::npos &&
//           ge::AttrUtils::HasAttr(node->GetOpDesc(), "_atomic_kernelname")) {
//         ++count;
//       }
//     }
//     EXPECT_EQ(count, 42);
//   };

  // // 校验context
  // std::ifstream infile;
  // std::string file_name = fe_env::FeEnvUtils::GetFFTSLogFile();
  // infile.open(file_name.c_str(), std::ios::in);
  // if (infile.is_open()) {
  //   char buf[512] = {0};
  //   while (infile.getline(buf, sizeof(buf))) {
  //     std::string line(buf);
  //     if (line.find("ready_context_num1") != std::string::npos) {
  //       EXPECT_EQ(line, "ready_context_num1=73");
  //     } else if (line.find("total_context_num1") != std::string::npos) {
  //       EXPECT_EQ(line, "total_context_num1=1703");
  //     } else if (line.find("prefetch_count1") != std::string::npos) {
  //       EXPECT_EQ(line, "prefetch_count1=0");
  //     } else if (line.find("invalid_count1") != std::string::npos) {
  //       EXPECT_EQ(line, "invalid_count1=699");
  //     } else if (line.find("writeback_count1") != std::string::npos) {
  //        EXPECT_EQ(line, "writeback_count1=205");
  //     }
  //   }
  //   infile.close();
  // }
  // remove(file_name.c_str());
// }

 /* 测试256batch下，编译时间优化。 ffts+切分，中间thread的算子如果和首切片的shape相同，
  * 不需要编译。 */
//  TEST_F(STestFeWholeProcess910B, test_resnet18_256batch_compilation_optimization) {
//    te::TeStub::GetInstance().SetImpl(std::make_shared<te::RealTimeCompilation>());
//    auto &fe_env = fe_env::FeRunningEnv::Instance();
//    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//    std::map<string, string> session_options;
//    session_options[ge::PRECISION_MODE] = fe::ALLOW_FP32_TO_FP16;
//    string network_path = fe_env::FeRunningEnv::GetNetworkPath("resnet18_256batch.txt");
//    bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//    ASSERT_EQ(state, true);
//    EXPECT_EQ(graph->GetName(), "ge_default_20230320160232");
//    EXPECT_EQ(graph->GetDirectNode().size(), 870);

//    //set runmode to ffts
//    ffts::Configuration::Instance().SetFftsOptimizeEnable();
//    /* 在执行前设置需要dump图的阶段 */
//    DUMP_GRAPH_WHEN("MergedComputeGraphAfterCompositeEnginePartition", "AfterGetTask");
//    EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//    CHECK_GRAPH(MergedComputeGraphAfterCompositeEnginePartition) {
//      // 校验切分的算子数
//      int32_t count = 0;
//      std::regex reg("FusedBatchNormV3.*_lxslice[0-9]+$");
//      for (auto &node : graph->GetAllNodes()) {
//        std::cmatch base_match;
//        if (std::regex_search(node->GetName().c_str(), base_match, reg)) {
//          ++count;
//        }
//      }
//      // EXPECT_EQ(count, 510);
//    };

//    CHECK_GRAPH(AfterGetTask) {
//      int32_t slice_count = 0;
//      // 校验atomic_clean_ptr
//      int32_t atomic_count = 0;
//      for (auto &node : graph->GetAllNodes()) {
//        if (node->GetName().find("FusedBatchNormV3") != string::npos &&
//            ge::AttrUtils::HasAttr(node->GetOpDesc(), "_atomic_kernelname")) {
//          ++atomic_count;
//        }
//        auto name = node->GetName();
//        if (name.find("loss_scale/gradients/fp32_vars/Relu_1_grad/ReluGrad_lxslice3") != string::npos ||
//            name.find("loss_scale/gradients/fp32_vars/conv2d/Conv2D_grad/Conv2DBackpropFilter_lxslice2") != string::npos ||
//            name.find("fp32_vars/BatchNorm_18/FusedBatchNormV3_Reduce_lxslice3") != string::npos ||
//            name.find("loss_scale/gradients/fp32_vars/conv2d_19/Conv2D_grad/Conv2DBackpropInput_lxslice8") != string::npos ||
//            name.find("fp32_vars/conv2d_18/Conv2D_lxslice9") != string::npos) {
//          // 检查算子被编译过，当前由于中间thread的算子跳过了编译，且使用的是额外属性，所以暂时只能校验是否有编译过；
//          // 在不加额外属性的情况下，不知道是否跳过了编译
//          bool has_compiled_result = node->GetOpDesc()->HasAttr(kKernelName);
//          bool no_need_compile = node->GetOpDesc()->HasAttr(kOriginalNode);
//          EXPECT_EQ(has_compiled_result, true);
//          EXPECT_EQ(no_need_compile, true);
//          slice_count++;
//        }
//      }

//      EXPECT_EQ(slice_count, 2);
//      EXPECT_EQ(atomic_count, 37);
//    };
//    unsetenv("DUMP_GE_GRAPH");
//  }

// TEST_F(STestFeWholeProcess910B, test_google_network) {
//   te::TeStub::GetInstance().SetImpl(std::make_shared<te::RealTimeCompilation>());
//   auto &fe_env = fe_env::FeRunningEnv::Instance();
//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//   std::map<string, string> session_options;
//   session_options[ge::PRECISION_MODE] = fe::ALLOW_FP32_TO_FP16;
//   session_options[ge::JIT_COMPILE] = "1";
//   string network_path = fe_env::FeRunningEnv::GetNetworkPath("google.txt");
//   bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//   ASSERT_EQ(state, true);
//   EXPECT_EQ(graph->GetName(), "ge_default_20230607092234");
//   EXPECT_EQ(graph->GetDirectNode().size(), 18);

//   // set runmode to ffts
//   ffts::Configuration::Instance().SetFftsOptimizeEnable();
//   DUMP_GRAPH_WHEN("MergedComputeGraphAfterCompositeEnginePartition", "PreRunAfterBuild");
//   EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);

//   CHECK_GRAPH(MergedComputeGraphAfterCompositeEnginePartition) {
//     // check the operator segmentation
//     int32_t count = 0;
//     std::regex reg("_lxslice[0-9]+$");
//     for (auto &node : graph->GetAllNodes()) {
//       std::cmatch base_match;
//       if (std::regex_search(node->GetName().c_str(), base_match, reg)) {
//         ++count;
//       }
//     }
//     EXPECT_EQ(count, 14);
//   };
//   CHECK_GRAPH(PreRunAfterBuild) {
//     int cnt = 0;
//     int net_output_cnt = 0;
//     for (auto &node : graph->GetAllNodes()) {
//       ++cnt;
//       if (node->GetType() == "NetOutput") {
//         ++net_output_cnt;
//       }
//     }
//     EXPECT_EQ(cnt, 44);
//     EXPECT_EQ(net_output_cnt, 2);
//   };

//   // expected generated task_def context
//   std::string task_def_path = fe_env::FeRunningEnv::GetNetworkPath("task_def_context.txt");
//   std::string task_def_str = fe_env::FeEnvUtils::GetTaskDefStr(task_def_path);
//   //std::cout << "task_def_str is "<< task_def_str;

//   // actual generated task_def context
//   std::string ascend_path = fe_env::FeEnvUtils::GetAscendPath();
//   std::string task_def_path1 = ascend_path + "/task_def_context.txt";
//   std::string task_def_str1 = fe_env::FeEnvUtils::GetTaskDefStr(task_def_path1);
//   //std::cout << "task_def_str1 is "<< task_def_str1;

//   bool same_size = task_def_str.size() == task_def_str1.size();
// //  EXPECT_EQ(same_size, true);
//   // newest lib it will failed, need check and reopen
//   //EXPECT_EQ(task_def_str1.compare(task_def_str), 0);
// }

//TEST_F(STestFeWholeProcess910B, test_support_unknown_shape) {
//  g_runtime_stub_mock = "rtMemcpyAsync";
//  te::TeStub::GetInstance().SetImpl(std::make_shared<te::RealTimeCompilation>());
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//  std::map<string, string> session_options;
//  session_options[ge::PRECISION_MODE] = fe::ALLOW_FP32_TO_FP16;
//  session_options[ge::JIT_COMPILE] = "1";
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("support_unknown_shape_1.txt");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//  EXPECT_EQ(graph->GetName(), "ge_default_20230329041545");
//  EXPECT_EQ(graph->GetDirectNode().size(), 9);
//
//  // set runmode to ffts
//  ffts::Configuration::Instance().SetFftsOptimizeEnable();
//  DUMP_GRAPH_WHEN("AfterDynamicShapePartition", "MergedComputeGraphAfterCompositeEnginePartition", "PreRunAfterBuild");
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//  int fusion_before_num = 0;
//  int fusion_after_num = 0;
//  CHECK_GRAPH(AfterDynamicShapePartition) {
//    for (auto &node : graph->GetAllNodes()) {
//      if (node->GetType() == "PartitionedCall") {
//        ++fusion_before_num;
//      }
//    }
//  };
//  CHECK_GRAPH(MergedComputeGraphAfterCompositeEnginePartition) {
//    for (auto &node : graph->GetAllNodes()) {
//      if (node->GetType() == "PartitionedCall") {
//        ++fusion_after_num;
//      }
//    }
//  };
//  CHECK_GRAPH(PreRunAfterBuild) {
//    int count = 0;
//    for (auto &node : graph->GetAllNodes()) {
//      ++count;
//    }
//    //EXPECT_EQ(count, 16);
//  };
//  //std::cout << "fusion_before_num is " << fusion_before_num << endl;
//  //std::cout << "fusion_after_num is " << fusion_after_num << endl;
//  //EXPECT_EQ(fusion_before_num, 2);
//  //EXPECT_EQ(fusion_before_num, fusion_after_num);
//}
//
//TEST_F(STestFeWholeProcess910B, test_random_truncated) {
//  g_runtime_stub_mock = "rtMemcpyAsync";
//  te::TeStub::GetInstance().SetImpl(std::make_shared<te::RealTimeCompilation>());
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//  std::map<string, string> session_options;
//  session_options[ge::PRECISION_MODE] = fe::ALLOW_FP32_TO_FP16;
//  session_options[ge::JIT_COMPILE] = "1";
//  session_options[ge::VIRTUAL_TYPE] = "0";
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("random_truncated.txt");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//  EXPECT_EQ(graph->GetName(), "ge_default_20230313235035");
//  EXPECT_EQ(graph->GetDirectNode().size(), 6);
//
//  // set runmode to ffts
//  ffts::Configuration::Instance().SetFftsOptimizeEnable();
//  DUMP_GRAPH_WHEN("PreRunAfterBuild");
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//  CHECK_GRAPH(PreRunAfterBuild) {
//    int count = 0;
//    for (auto &node : graph->GetAllNodes()) {
//      if (node->GetType() == fe::CONSTANT || node->GetType() == fe::CONSTANTOP) {
//        bool flag = false;
//        std::shared_ptr<const ge::GeTensor> tensor = nullptr;
//        if (ge::AttrUtils::GetTensor(node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, tensor)) {
//          flag = true;
//          EXPECT_NE(tensor, nullptr);
//          EXPECT_EQ(*(tensor->GetData().GetData()), 0);
//        }
//        EXPECT_NE(flag, true);
//      }
//      ++count;
//    }
//    EXPECT_EQ(count, 7);
//  };
//}
//
//TEST_F(STestFeWholeProcess910B, test_ub_fusion) {
//  g_runtime_stub_mock = "rtMemcpyAsync";
//  te::TeStub::GetInstance().SetImpl(std::make_shared<te::RealTimeCompilation>());
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//  std::map<string, string> session_options;
//  session_options[ge::PRECISION_MODE] = fe::ALLOW_FP32_TO_FP16;
//  session_options[ge::JIT_COMPILE] = "1";
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("fusion.txt");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//  EXPECT_EQ(graph->GetName(), "ub_fusion_common_rules2_002");
//  EXPECT_EQ(graph->GetDirectNode().size(), 21);
//
//  // set runmode to ffts
//  ffts::Configuration::Instance().SetFftsOptimizeEnable();
//  DUMP_GRAPH_WHEN("PreRunAfterBuild");
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//  CHECK_GRAPH(PreRunAfterBuild) {
//    int count = 0;
//    for (auto &node : graph->GetAllNodes()) {
//      if (node->GetType() == "Relu6") {
//        //std::cout << "name is" << node->GetName().c_str()<<endl;
//        string pass_name_ub;
//        (void)ge::AttrUtils::GetStr(node->GetOpDesc(), "pass_name_ub", pass_name_ub);
//        EXPECT_EQ(pass_name_ub, "TbeMultiOutputFusionPass");
//      }
//      ++count;
//    }
//    EXPECT_EQ(count, 14);
//  };
//}
//
//TEST_F(STestFeWholeProcess910B, test_quant_split_optimize) {
//  g_runtime_stub_mock = "rtMemcpyAsync";
//  te::TeStub::GetInstance().SetImpl(std::make_shared<te::RealTimeCompilation>());
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//  std::map<string, string> session_options;
//  session_options[ge::PRECISION_MODE] = fe::ALLOW_FP32_TO_FP16;
//  session_options[ge::JIT_COMPILE] = "1";
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("tf_quant_split_optimize_c.txt");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//  EXPECT_EQ(graph->GetName(), "tf_quant_split_optimize_c_101");
//  EXPECT_EQ(graph->GetDirectNode().size(), 33);
//
//  // set runmode to ffts
//  ffts::Configuration::Instance().SetFftsOptimizeEnable();
//  DUMP_GRAPH_WHEN("PreRunAfterOptimizeGraphBeforeBuild", "PreRunAfterBuild");
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//  CHECK_GRAPH(PreRunAfterOptimizeGraphBeforeBuild) {
//    for (auto &node : graph->GetAllNodes()) {
//      if (node->GetType() == "PhonyConcat") {
//        bool is_input_continuous = false;
//        (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "_no_padding_continuous_input", is_input_continuous);
//        bool attr_notask = false;
//        (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "_no_task", attr_notask);
//        EXPECT_EQ(is_input_continuous, true);
//        EXPECT_EQ(attr_notask, true);
//      }
//    }
//  };
//  CHECK_GRAPH(PreRunAfterBuild) {
//    int count = 0;
//    bool flag = false;
//    for (auto &node : graph->GetAllNodes()) {
//      if (node->GetType() == "ConcatV2D" || node->GetType() == "SplitD") {
//        flag = true;
//      }
//      ++count;
//    }
//    EXPECT_EQ(count, 25);
//    EXPECT_EQ(flag, true);
//  };
//}
//
//TEST_F(STestFeWholeProcess910B, test_allowFp32toFp16_select_forcefp16_014) {
//  g_runtime_stub_mock = "rtMemcpyAsync";
//  te::TeStub::GetInstance().SetImpl(std::make_shared<te::RealTimeCompilation>());
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//  std::map<string, string> session_options;
//  session_options[ge::PRECISION_MODE] = fe::ALLOW_FP32_TO_FP16;
//  session_options[ge::JIT_COMPILE] = "1";
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("allowFp32toFp16_select_forcefp16_014.txt");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//  EXPECT_EQ(graph->GetName(), "allowFp32toFp16_select_forcefp16_014");
//  EXPECT_EQ(graph->GetDirectNode().size(), 3);
//
//  // set runmode to ffts
//  ffts::Configuration::Instance().SetFftsOptimizeEnable();
//  DUMP_GRAPH_WHEN("PreRunAfterBuild");
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//  CHECK_GRAPH(PreRunAfterBuild) {
//    // check ubfusion pass name
//    std::string pass_name;
//    for (auto &node : graph->GetAllNodes()) {
//      if (node->GetName().find("conv2d_pass.1") != string::npos) {
//        (void)ge::AttrUtils::GetStr(node->GetOpDesc(), "pass_name_ub", pass_name);
//      }
//    }
//    EXPECT_EQ(pass_name, "TbeFixPipeFusionPass");
//  };
//}
//TEST_F(STestFeWholeProcess910B, test_allow_hf32_op_impl_mode) {
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//  std::map<string, string> session_options;
//  session_options["ge.exec.allow_hf32"] = "10";
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("conv2d.txt");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//
//  DUMP_GRAPH_WHEN("OptimizeQuantGraph_FeGraphFusionAfter");
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//  CHECK_GRAPH(OptimizeQuantGraph_FeGraphFusionAfter) {
//    for (auto &node : graph->GetDirectNode()) {
//      if (node->GetType().compare("Conv2D") == 0) {
//        int64_t op_impl_mode = 0;
//        ge::AttrUtils::GetInt(node->GetOpDesc(), OP_IMPL_MODE_ENUM, op_impl_mode);
//        EXPECT_EQ(op_impl_mode, 0x40);
//      }
//    }
//  };
//}
}
