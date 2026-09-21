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
#include <nlohmann/json.hpp>
#define protected public
#define private public
#include <graph/tensor.h>
#include "framework/fe_running_env/include/fe_running_env.h"
#include "graph_optimizer/fe_graph_optimizer.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph_constructor/graph_constructor.h"
#include "register/ops_kernel_builder_registry.h"
#include "st_whole_process/stub/aicore_ops_kernel_builder_stub.h"
#include "google/protobuf/text_format.h"
#undef protected
#undef private
#include "stub/te_fusion_api.h"
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;
using CreateFn = ge::OpsKernelBuilder *(*)();
using OpsKernelBuilderPtr = std::shared_ptr<ge::OpsKernelBuilder>;
namespace fe {
class STestFeWholeProcess910A: public testing::Test
{
 protected:
  static void SetUpTestCase() {
    const char *soc_version = "Ascend910A";
    rtSetSocVersion(soc_version);
    auto &fe_env = fe_env::FeRunningEnv::Instance();
    std::map<string, string> new_options;
    new_options.emplace(ge::SOC_VERSION, soc_version);
    fe_env.InitRuningEnv(new_options);
  }

  static void TearDownTestCase() {
    te::TeStub::GetInstance().SetImpl(std::make_shared<te::TeStubApi>());
    auto &fe_env = fe_env::FeRunningEnv::Instance();
    fe_env.FinalizeEnv();
  }
};

void ConvertToStr(string &task_def_str) {
  for (auto &task_iter : fe_env::FeRunningEnv::tasks_map) {
    for (auto &task : task_iter.second) {
      std::string task_def_str_tmp;
      google::protobuf::TextFormat::PrintToString(task, &task_def_str_tmp);
      task_def_str.append(task_def_str_tmp);
    }
  }
}

// TEST_F(STestFeWholeProcess910A, test_tasks_check) {
//   /* This case uses real compilation environment. */
//   te::TeStub::GetInstance().SetImpl(std::make_shared<te::RealTimeCompilation>());
//   auto &fe_env = fe_env::FeRunningEnv::Instance();
//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//   std::map<string, string> session_options;
//   session_options[ge::PRECISION_MODE] = fe::FORCE_FP16;

//   string network_path = fe_env::FeRunningEnv::GetNetworkPath("inceptionv3_aipp_int8_16batch.txt");
//   string task_context_path = fe_env::FeRunningEnv::GetNetworkPath("task_context.txt");
//   bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//   ASSERT_EQ(state, true);
//   ASSERT_EQ(graph->GetName(), "inceptionv3_framework_caffe_aipp_1_batch_16_input_int8_output_FP32");
//   EXPECT_EQ(graph->GetDirectNode().size(), 661);

//   const auto &ops_kernel_builders_const = ge::OpsKernelBuilderRegistry::GetInstance().GetAll();
//   std::map<std::string, OpsKernelBuilderPtr> &ops_kernel_builders = const_cast<std::map<std::string, OpsKernelBuilderPtr>&>(ops_kernel_builders_const);

//   for (auto &ops_kernel_builder: ops_kernel_builders) {
//     if (ops_kernel_builder.first == AI_CORE_NAME) {
//       AICoreOpsKernelSubBuilder test;
//       CreateFn fn = []()->::ge::OpsKernelBuilder* {return new (std::nothrow) (AICoreOpsKernelSubBuilder)();};
//       ops_kernel_builder.second.reset(fn());
//       std::cout << "inject AICoreOpsKernelSubBuilder success!" << std::endl;
//     }
//   }

//   auto ret = fe_env.Run(graph, session_options);
//   EXPECT_EQ(ret, fe::SUCCESS);

//   std::string task_def_str;
//   ConvertToStr(task_def_str);

//   ofstream ofs;
//   ofs.open("./task_context.txt", ios::out);
//   if (!ofs.is_open()) {
//     std::cout << "open to write task_context failed!" << std::endl;
// 	}
//   ofs << task_def_str;
//   ofs.close();
// }

// TEST_F(STestFeWholeProcess910A, test_inceptionv3_aipp_int8_16batch) {
//   /* This case uses real compilation environment. */
//   te::TeStub::GetInstance().SetImpl(std::make_shared<te::RealTimeCompilation>());
//   auto &fe_env = fe_env::FeRunningEnv::Instance();
//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//   std::map<string, string> session_options;
//   session_options[ge::PRECISION_MODE] = fe::FORCE_FP16;

//   string network_path = fe_env::FeRunningEnv::GetNetworkPath("inceptionv3_aipp_int8_16batch.txt");
//   bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//   ASSERT_EQ(state, true);
//   ASSERT_EQ(graph->GetName(), "inceptionv3_framework_caffe_aipp_1_batch_16_input_int8_output_FP32");
//   EXPECT_EQ(graph->GetDirectNode().size(), 661);
//   /* 在执行前设置需要dump图的阶段 */
//   DUMP_GRAPH_WHEN("OptimizeOriginalGraph_FeInsertTransNodeAfter", "PreRunAfterBuild");
//   EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);

//   /* 在执行后可以对特定阶段的图进行校验 */
//   CHECK_GRAPH(OptimizeOriginalGraph_FeInsertTransNodeAfter) {
//     EXPECT_EQ(graph->GetDirectNodesSize(), 1257);
//     size_t trans_data_size = 0; //191
//     size_t requant_host_cpu_size = 0; // 93
//     size_t quant_bias_optimization_size = 0; //85 QuantBiasOptimization
//     size_t quant_size = 0; // 84 AscendQuant
//     size_t conv2d_size = 0; // 90 Conv2D
//     size_t pooling_size = 0; // 14 Pooling
//     for (auto &node : graph->GetDirectNode()) {
//       string type = node->GetType();
//       if (type == "TransData") {
//         trans_data_size++;
//       } else if (type == "RequantHostCpuOp") {
//         requant_host_cpu_size++;
//       } else if (type == "QuantBiasOptimization") {
//         quant_bias_optimization_size++;
//       } else if (type == "AscendQuant"){
//         quant_size++;
//       } else if (type == "Conv2D") {
//         conv2d_size++;
//       } else if (type == "Pooling") {
//         pooling_size++;
//       }
//     }
//     EXPECT_EQ(trans_data_size, 191);
//     EXPECT_EQ(requant_host_cpu_size, 93);
//     EXPECT_EQ(quant_bias_optimization_size, 85);
//     EXPECT_EQ(quant_size, 84);
//     EXPECT_EQ(conv2d_size, 90);
//     EXPECT_EQ(pooling_size, 14);
//   };


//   CHECK_GRAPH(PreRunAfterBuild) {
//     //std::cout << "PreRunAfterBuild GetDirectNodesSize = " <<  graph->GetDirectNodesSize() << std::endl;
//     size_t trans_data_size = 0; //192
//     size_t requant_host_cpu_size = 0; // 93
//     size_t quant_bias_optimization_size = 0; //85 QuantBiasOptimization
//     size_t quant_size = 0; // 84 AscendQuant
//     size_t conv2d_size = 0; // 90 Conv2D
//     size_t pooling_size = 0; // 14 Pooling
//     for (auto &node : graph->GetDirectNode()) {
//       string val;
//       if (node->GetName() == "conv1_3x3_s2conv1_3x3_reluconv2_3x3_s1_0_quant_layer") {
//         (void)ge::AttrUtils::GetStr(node->GetOpDesc(), kAttrKernelBinId, val);
//         EXPECT_NE(val, "");
//       }
//       string type = node->GetType();
//       if (type == "TransData") {
//         trans_data_size++;
//       } else if (type == "RequantHostCpuOp") {
//         requant_host_cpu_size++;
//       } else if (type == "QuantBiasOptimization") {
//         quant_bias_optimization_size++;
//       } else if (type == "AscendQuant"){
//         quant_size++;
//       } else if (type == "Conv2D") {
//         conv2d_size++;
//       } else if (type == "Pooling") {
//         pooling_size++;
//       }
//     }
//     EXPECT_EQ(trans_data_size, 2);
//     EXPECT_EQ(requant_host_cpu_size, 0);
//     EXPECT_EQ(quant_bias_optimization_size, 0);
//     EXPECT_EQ(quant_size, 4);
//     EXPECT_EQ(conv2d_size, 82);
//     EXPECT_EQ(pooling_size, 11);
//   };
// }

// TEST_F(STestFeWholeProcess910A, test_mul_single_op_fp32) {
//   te::TeStub::GetInstance().SetImpl(std::make_shared<te::TeStubApi>());
//   auto &fe_env = fe_env::FeRunningEnv::Instance();
//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//   ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
//   GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

//   test.AddOpDesc("Data_0", "Data", 1, 1)
//       .AddOpDesc("Data_1", "Data", 1, 1)
//       .SetInput("Mul_0:0", "Data_0")
//       .SetInput("Mul_0:1", "Data_1")
//       .SetInput("NetOutput:0", "Mul_0:0");
//   std::map<string, string> session_options;
//   EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//   EXPECT_EQ(graph->GetDirectNode().size(), 4);
// }

// TEST_F(STestFeWholeProcess910A, test_binary_reuse) {
//   auto &fe_env = fe_env::FeRunningEnv::Instance();
//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//   string network_path = fe_env::FeRunningEnv::GetNetworkPath("identity.txt");
//   bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//   ASSERT_EQ(state, true);
//   EXPECT_EQ(graph->GetName(), "20221229201611_online");
//   EXPECT_EQ(graph->GetDirectNode().size(), 3);
//   std::map<string, string> session_options;
//   session_options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", "shape_generalized"));
//   ge::GetThreadLocalContext().SetGlobalOption(session_options);
//   DUMP_GRAPH_WHEN("PreRunAfterInitPreparation", "OptimizeGraph_TagNoConstFoldingAfter");
//   EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);

//   CHECK_GRAPH(PreRunAfterInitPreparation) {
//     EXPECT_EQ(graph->GetDirectNodesSize(), 3);
//     for (auto &node : graph->GetDirectNode()) {
//       if (node->GetName() == "Identity") {
//         auto op = node->GetOpDesc();
//         auto input = op->MutableInputDesc(0);
//         ASSERT_NE(input, nullptr);
//         EXPECT_EQ(input->GetFormat(), ge::FORMAT_NC1HWC0);
//         EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(input->GetShape(), ge::GeShape({64, 3, 7, 7}));
//         auto output = op->MutableOutputDesc(0);
//         ASSERT_NE(output, nullptr);
//         EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
//         EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(output->GetShape(), ge::GeShape({64, 3, 7, 7}));
//         EXPECT_EQ(output->GetOriginShape(), ge::GeShape({1, 3, 7, 7}));
//       }
//     }
//   };
//   CHECK_GRAPH(OptimizeGraph_TagNoConstFoldingAfter) {
//     EXPECT_EQ(graph->GetDirectNodesSize(), 3);
//     for (auto &node : graph->GetDirectNode()) {
//       if (node->GetName() == "Identity") {
//         auto op = node->GetOpDesc();
//         auto input = op->MutableInputDesc(0);
//         ASSERT_NE(input, nullptr);
//         EXPECT_EQ(input->GetFormat(), ge::FORMAT_NC1HWC0);
//         EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(input->GetShape(), ge::GeShape({-1, -1, -1, -1}));
//         auto output = op->MutableOutputDesc(0);
//         ASSERT_NE(output, nullptr);
//         EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
//         EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(output->GetShape(), ge::GeShape({64, 3, 7, 7}));
//         EXPECT_EQ(output->GetOriginShape(), ge::GeShape({1, 3, 7, 7}));
//       }
//     }
//   };
// }

// TEST_F(STestFeWholeProcess910A, test_binary_reuse_1) {
//   auto &fe_env = fe_env::FeRunningEnv::Instance();
//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//   string network_path = fe_env::FeRunningEnv::GetNetworkPath("broadcastto.txt");
//   bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//   ASSERT_EQ(state, true);
//   EXPECT_EQ(graph->GetName(), "20230117112206_op_models");
//   EXPECT_EQ(graph->GetDirectNode().size(), 4);
//   std::map<string, string> session_options;
//   session_options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", "shape_generalized"));
//   session_options.insert(std::pair<std::string, std::string>("ge.shape_generalized", "1"));
//   ge::GetThreadLocalContext().SetGlobalOption(session_options);
//   DUMP_GRAPH_WHEN("PreRunAfterInitPreparation", "OptimizeGraph_TagNoConstFoldingAfter");
//   EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);

//   CHECK_GRAPH(PreRunAfterInitPreparation) {
//     EXPECT_EQ(graph->GetDirectNodesSize(), 4);
//     for (auto &node : graph->GetDirectNode()) {
//       if (node->GetName() == "BroadcastTo") {
//         auto op = node->GetOpDesc();
//         auto input = op->MutableInputDesc(0);
//         ASSERT_NE(input, nullptr);
//         EXPECT_EQ(input->GetFormat(), ge::FORMAT_ND);
//         EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT16);
//         EXPECT_EQ(input->GetShape(), ge::GeShape({16}));
//         auto input1 = op->MutableInputDesc(1);
//         ASSERT_NE(input1, nullptr);
//         EXPECT_EQ(input1->GetFormat(), ge::FORMAT_ND);
//         EXPECT_EQ(input1->GetDataType(), ge::DT_INT32);
//         EXPECT_EQ(input1->GetShape(), ge::GeShape({1}));
//         auto output = op->MutableOutputDesc(0);
//         ASSERT_NE(output, nullptr);
//         EXPECT_EQ(output->GetFormat(), ge::FORMAT_ND);
//         EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT16);
//         EXPECT_EQ(output->GetShape(), ge::GeShape({16}));
//         EXPECT_EQ(output->GetOriginShape(), ge::GeShape({16}));
//       }
//     }
//   };
//   CHECK_GRAPH(OptimizeGraph_TagNoConstFoldingAfter) {
//     EXPECT_EQ(graph->GetDirectNodesSize(), 4);
//     for (auto &node : graph->GetDirectNode()) {
//       if (node->GetName() == "BroadcastTo") {
//         auto op = node->GetOpDesc();
//         auto input = op->MutableInputDesc(0);
//         ASSERT_NE(input, nullptr);
//         EXPECT_EQ(input->GetFormat(), ge::FORMAT_ND);
//         EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT16);
//         EXPECT_EQ(input->GetShape(), ge::GeShape({-2}));
//         EXPECT_EQ(input->GetOriginShape(), ge::GeShape({-2}));
//         auto input1 = op->MutableInputDesc(1);
//         ASSERT_NE(input1, nullptr);
//         EXPECT_EQ(input1->GetFormat(), ge::FORMAT_ND);
//         EXPECT_EQ(input1->GetDataType(), ge::DT_INT32);
//         EXPECT_EQ(input1->GetShape(), ge::GeShape({1}));
//         auto output = op->MutableOutputDesc(0);
//         ASSERT_NE(output, nullptr);
//         EXPECT_EQ(output->GetFormat(), ge::FORMAT_ND);
//         EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT16);
//         EXPECT_EQ(output->GetShape(), ge::GeShape({16}));
//         EXPECT_EQ(output->GetOriginShape(), ge::GeShape({16}));
//       }
//     }
//   };
// }

// TEST_F(STestFeWholeProcess910A, test_binary_reuse_2) {
//   auto &fe_env = fe_env::FeRunningEnv::Instance();
//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//   string network_path = fe_env::FeRunningEnv::GetNetworkPath("conv2d.txt");
//   bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//   ASSERT_EQ(state, true);
//   EXPECT_EQ(graph->GetName(), "ge_default_20230118103502");
//   EXPECT_EQ(graph->GetDirectNode().size(), 6);
//   std::map<string, string> session_options;
//   session_options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", "shape_generalized"));
//   session_options.insert(std::pair<std::string, std::string>("ge.shape_generalized", "1"));
//   ge::GetThreadLocalContext().SetGlobalOption(session_options);
//   DUMP_GRAPH_WHEN("PreRunAfterInitPreparation", "OptimizeGraph_TagNoConstFoldingAfter");
//   EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//   CHECK_GRAPH(PreRunAfterInitPreparation) {
//     EXPECT_EQ(graph->GetDirectNodesSize(), 6);
//     for (auto &node : graph->GetDirectNode()) {
//       if (node->GetName() == "Conv2D") {
//         auto op = node->GetOpDesc();
//         auto input = op->MutableInputDesc(0);
//         ASSERT_NE(input, nullptr);
//         EXPECT_EQ(input->GetFormat(), ge::FORMAT_NHWC);
//         EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(input->GetShape(), ge::GeShape({3, 3, 3, 3}));
//         EXPECT_EQ(input->GetOriginShape(), ge::GeShape({3, 3, 3, 3}));
//         auto input1 = op->MutableInputDesc(1);
//         ASSERT_NE(input1, nullptr);
//         EXPECT_EQ(input1->GetFormat(), ge::FORMAT_HWCN);
//         EXPECT_EQ(input1->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(input1->GetShape(), ge::GeShape({3, 3, 3, 3}));
//         EXPECT_EQ(input1->GetOriginShape(), ge::GeShape({3, 3, 3, 3}));
//         auto output = op->MutableOutputDesc(0);
//         ASSERT_NE(output, nullptr);
//         EXPECT_EQ(output->GetFormat(), ge::FORMAT_NHWC);
//         EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(output->GetShape(), ge::GeShape({3, 1, 1, 3}));
//         EXPECT_EQ(output->GetOriginShape(), ge::GeShape({3, 1, 1, 3}));
//       }
//     }
//   };
//   CHECK_GRAPH(OptimizeGraph_TagNoConstFoldingAfter) {
//     EXPECT_EQ(graph->GetDirectNodesSize(), 6);
//     for (auto &node : graph->GetDirectNode()) {
//       if (node->GetName() == "Conv2D") {
//         auto op = node->GetOpDesc();
//         auto input = op->MutableInputDesc(0);
//         ASSERT_NE(input, nullptr);
//         EXPECT_EQ(input->GetFormat(), ge::FORMAT_NHWC);
//         EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(input->GetShape(), ge::GeShape({-1, -1, -1, -1}));
//         EXPECT_EQ(input->GetOriginShape(), ge::GeShape({-1, -1, -1, -1}));
//         auto input1 = op->MutableInputDesc(1);
//         ASSERT_NE(input1, nullptr);
//         EXPECT_EQ(input1->GetFormat(), ge::FORMAT_HWCN);
//         EXPECT_EQ(input1->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(input1->GetShape(), ge::GeShape({3, 3, 3, 3}));
//         EXPECT_EQ(input1->GetOriginShape(), ge::GeShape({3, 3, 3, 3}));
//         auto output = op->MutableOutputDesc(0);
//         ASSERT_NE(output, nullptr);
//         EXPECT_EQ(output->GetFormat(), ge::FORMAT_NHWC);
//         EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(output->GetShape(), ge::GeShape({3, 1, 1, 3}));
//         EXPECT_EQ(output->GetOriginShape(), ge::GeShape({3, 1, 1, 3}));
//       }
//     }
//   };
// }

// TEST_F(STestFeWholeProcess910A, test_npuallocfloatstatus) {
//   auto &fe_env = fe_env::FeRunningEnv::Instance();
//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//   string network_path = fe_env::FeRunningEnv::GetNetworkPath("npuallocfloatstatus.txt");
//   bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//   ASSERT_EQ(state, true);
//   EXPECT_EQ(graph->GetName(), "20230215183926_online");
//   EXPECT_EQ(graph->GetDirectNode().size(), 2);
//   std::map<string, string> session_options;
//   session_options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", "shape_generalized"));
//   session_options.insert(std::pair<std::string, std::string>("ge.shape_generalized", "1"));
//   ge::GetThreadLocalContext().SetGlobalOption(session_options);
//   DUMP_GRAPH_WHEN("PreRunAfterInitPreparation", "OptimizeGraph_TagNoConstFoldingAfter");
//   EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//   // fe_env.Run(graph, session_options);
// }

TEST_F(STestFeWholeProcess910A, test_conv3dbackpropfilter_generalize) {
  auto &fe_env = fe_env::FeRunningEnv::Instance();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
  string network_path = fe_env::FeRunningEnv::GetNetworkPath("conv3dbackpropfilter.txt");
  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
  ASSERT_EQ(state, true);
  std::map<string, string> session_options;
  session_options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", "shape_generalized"));
  session_options.insert(std::pair<std::string, std::string>("ge.shape_generalized", "1"));
  ge::GetThreadLocalContext().SetGlobalOption(session_options);
//  DUMP_GRAPH_WHEN("OptimizeGraph_TagNoConstFoldingAfter", "BeforeOptimizeOriginalGraphJudgeInsert");
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//  CHECK_GRAPH(OptimizeGraph_TagNoConstFoldingAfter) {
//    EXPECT_EQ(graph->GetDirectNodesSize(), 5);
//    for (auto &node : graph->GetDirectNode()) {
//      if (node->GetName() == "Conv3DBackpropFilter") {
//        auto op = node->GetOpDesc();
//        auto input = op->MutableInputDesc(0);
//        ASSERT_NE(input, nullptr);
//        EXPECT_EQ(input->GetShape(), ge::GeShape({-1, 512, -1, -1, -1}));
//        EXPECT_EQ(input->GetOriginShape(), ge::GeShape({-1, 512, -1, -1, -1}));
//        std::vector<std::pair<int64_t, int64_t>> range({{32, 2147483647}, {512, 512}, {1, 3}, {4, 15}, {4, 15}});
//        std::vector<std::pair<int64_t, int64_t>> shape_range;
//        (void)input->GetShapeRange(shape_range);
//        EXPECT_EQ(range, shape_range);
//        auto input1 = op->MutableInputDesc(1);
//        EXPECT_EQ(input1->GetShape(), ge::GeShape({64, 512, 2, 7, 7}));
//        EXPECT_EQ(input1->GetOriginShape(), ge::GeShape({64, 512, 2, 7, 7}));
//        std::vector<std::pair<int64_t, int64_t>> range1;
//        std::vector<std::pair<int64_t, int64_t>> shape_range1;
//        (void)input1->GetShapeRange(shape_range1);
//        EXPECT_EQ(range1, shape_range1);
//        auto input2 = op->MutableInputDesc(2);
//        EXPECT_EQ(input2->GetShape(), ge::GeShape({64, 512, 2, 7, 7}));
//        EXPECT_EQ(input2->GetOriginShape(), ge::GeShape({64, 512, 2, 7, 7}));
//        std::vector<std::pair<int64_t, int64_t>> range2;
//        std::vector<std::pair<int64_t, int64_t>> shape_range2;
//        (void)input2->GetShapeRange(shape_range);
//        EXPECT_EQ(range2, shape_range2);
//      }
//    }
//  };
//
//  CHECK_GRAPH(BeforeOptimizeOriginalGraphJudgeInsert) {
//    EXPECT_EQ(graph->GetDirectNodesSize(), 5);
//    for (auto &node : graph->GetDirectNode()) {
//      if (node->GetName() == "Conv3DBackpropFilter") {
//        auto op = node->GetOpDesc();
//        auto input = op->MutableInputDesc(0);
//        ASSERT_NE(input, nullptr);
//        EXPECT_EQ(input->GetShape(), ge::GeShape({-1, 512, -1, -1, -1}));
//        EXPECT_EQ(input->GetOriginShape(), ge::GeShape({-1, 512, -1, -1, -1}));
//        std::vector<std::pair<int64_t, int64_t>> range({{32, 2147483647}, {512, 512}, {1, 3}, {4, 15}, {4, 15}});
//        std::vector<std::pair<int64_t, int64_t>> shape_range;
//        (void)input->GetShapeRange(shape_range);
//        EXPECT_EQ(range, shape_range);
//        auto input1 = op->MutableInputDesc(1);
//        EXPECT_EQ(input1->GetShape(), ge::GeShape({64, 512, 2, 7, 7}));
//        EXPECT_EQ(input1->GetOriginShape(), ge::GeShape({64, 512, 2, 7, 7}));
//        std::vector<std::pair<int64_t, int64_t>> range1;
//        std::vector<std::pair<int64_t, int64_t>> shape_range1;
//        (void)input1->GetShapeRange(shape_range1);
//        EXPECT_EQ(range1, shape_range1);
//        auto input2 = op->MutableInputDesc(2);
//        EXPECT_EQ(input2->GetShape(), ge::GeShape({64, 512, 2, 7, 7}));
//        EXPECT_EQ(input2->GetOriginShape(), ge::GeShape({64, 512, 2, 7, 7}));
//        std::vector<std::pair<int64_t, int64_t>> range2;
//        std::vector<std::pair<int64_t, int64_t>> shape_range2;
//        (void)input2->GetShapeRange(shape_range);
//        EXPECT_EQ(range2, shape_range2);
//      }
//    }
//  };
}

// TEST_F(STestFeWholeProcess910A, test_dynamic_rnn) {
//   auto &fe_env = fe_env::FeRunningEnv::Instance();
//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//   string network_path = fe_env::FeRunningEnv::GetNetworkPath("dynamic_rnn.txt");
//   bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//   ASSERT_EQ(state, true);

//   DUMP_GRAPH_WHEN("OptimizeOriginalGraph_FeGraphFusionAfter", "PreRunAfterBuild");
//   std::map<string, string> session_options;
//   EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//   CHECK_GRAPH(OptimizeOriginalGraph_FeGraphFusionAfter) {
//     for (auto &node : graph->GetDirectNode()) {
//       if (node->GetType() == "DynamicRNN") {
//         auto op = node->GetOpDesc();
//         auto input = op->MutableInputDesc(0);
//         ASSERT_NE(input, nullptr);
//         EXPECT_EQ(input->GetFormat(), ge::FORMAT_NCHW);
//         EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(input->GetShape(), ge::GeShape({2, 64, 256}));
//         auto input1 = op->MutableInputDesc(1);
//         ASSERT_NE(input1, nullptr);
//         EXPECT_EQ(input1->GetFormat(), ge::FORMAT_ND);
//         EXPECT_EQ(input1->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(input1->GetShape(), ge::GeShape({512, 1024}));
//         auto input2 = op->MutableInputDesc(2);
//         ASSERT_NE(input2, nullptr);
//         EXPECT_EQ(input2->GetFormat(), ge::FORMAT_ND);
//         EXPECT_EQ(input2->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(input2->GetShape(), ge::GeShape({1024}));
//         auto input4 = op->MutableInputDesc(4);
//         ASSERT_NE(input4, nullptr);
//         EXPECT_EQ(input4->GetFormat(), ge::FORMAT_NCHW);
//         EXPECT_EQ(input4->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(input4->GetShape(), ge::GeShape({1, 64, 256}));
//         auto input5 = op->MutableInputDesc(5);
//         ASSERT_NE(input5, nullptr);
//         EXPECT_EQ(input5->GetFormat(), ge::FORMAT_NCHW);
//         EXPECT_EQ(input5->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(input5->GetShape(), ge::GeShape({1, 64, 256}));
//         auto output = op->MutableOutputDesc(0);
//         ASSERT_NE(output, nullptr);
//         EXPECT_EQ(output->GetFormat(), ge::FORMAT_NCHW);
//         EXPECT_EQ(output->GetDataType(), ge::DT_FLOAT);
//         EXPECT_EQ(output->GetShape(), ge::GeShape({2, 64, 256}));
//       }
//     }
//   };
//   CHECK_GRAPH(PreRunAfterBuild) {
//     EXPECT_EQ(graph->GetDirectNodesSize(), 16);
//     size_t trans_data_size = 0; //192
//     for (auto &node : graph->GetDirectNode()) {
//       string type = node->GetType();
//       if (type == "TransData") {
//         trans_data_size++;
//       }
//       if (node->GetType() == "DynamicRNN") {
//         auto op = node->GetOpDesc();
//         auto input = op->MutableInputDesc(0);
//         ASSERT_NE(input, nullptr);
//         EXPECT_EQ(ge::GetPrimaryFormat(input->GetFormat()), ge::FORMAT_FRACTAL_NZ);
//         auto input1 = op->MutableInputDesc(1);
//         ASSERT_NE(input1, nullptr);
//         EXPECT_EQ(ge::GetPrimaryFormat(input1->GetFormat()), ge::FORMAT_FRACTAL_ZN_RNN);
//         auto input2 = op->MutableInputDesc(2);
//         ASSERT_NE(input2, nullptr);
//         EXPECT_EQ(ge::GetPrimaryFormat(input2->GetFormat()), ge::FORMAT_ND_RNN_BIAS);
//         auto input4 = op->MutableInputDesc(4);
//         ASSERT_NE(input4, nullptr);
//         EXPECT_EQ(ge::GetPrimaryFormat(input4->GetFormat()), ge::FORMAT_FRACTAL_NZ);
//         auto input5 = op->MutableInputDesc(5);
//         ASSERT_NE(input5, nullptr);
//         EXPECT_EQ(ge::GetPrimaryFormat(input5->GetFormat()), ge::FORMAT_FRACTAL_NZ);
//         auto output = op->MutableOutputDesc(0);
//         ASSERT_NE(output, nullptr);
//         EXPECT_EQ(ge::GetPrimaryFormat(output->GetFormat()), ge::FORMAT_FRACTAL_NZ);
//       }
//     }
//     EXPECT_EQ(trans_data_size, 4);
//   };
// }
}
