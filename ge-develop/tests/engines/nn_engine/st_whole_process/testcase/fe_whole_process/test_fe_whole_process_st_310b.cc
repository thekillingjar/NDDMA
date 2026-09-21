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
namespace fe {
class STestFeWholeProcess310B: public testing::Test
{
 protected:
  void SetUp() {
    if (reset_count_ == 0) {
      const char *soc_version = "Ascend310B";
      rtSetSocVersion(soc_version);
      auto &fe_env = fe_env::FeRunningEnv::Instance();
      std::map<string, string> new_options;
      new_options[ge::AICORE_NUM] = "1";
      fe_env.InitRuningEnv(new_options);
      reset_count_++;
    }
  }

  void TearDown() {
    te::TeStub::GetInstance().SetImpl(std::make_shared<te::TeStubApi>());
    auto &fe_env = fe_env::FeRunningEnv::Instance();
    fe_env.FinalizeEnv();
  }
  int32_t reset_count_ = 0;
};

//TEST_F(STestFeWholeProcess310B, test_ub_fusion_cube_quant_requant) {
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("ub_fusion_cube_quant_requant.txt");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//  EXPECT_EQ(graph->GetName(), "ub_fusion_cube_quant_requant_010");
//  EXPECT_EQ(graph->GetDirectNode().size(), 19);
//  std::map<string, string> session_options;
//  DUMP_GRAPH_WHEN("PreRunAfterBuild");
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//  CHECK_GRAPH(PreRunAfterBuild) {
//    for (auto &node : graph->GetAllNodes()) {
//      if (node->GetType() == "LeakyRelu") {
//        auto const_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
//        auto input_node = const_anchor->GetOwnerNode();
//        EXPECT_NE(input_node, nullptr);
//      }
//    }
//  };
//}
//
//TEST_F(STestFeWholeProcess310B, test_ascend_quant_sqrt_mode) {
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("ascend_quant_sqrt_mode_test.txt");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//  EXPECT_EQ(graph->GetName(), "ascend_quant_sqrt_mode_test");
//  EXPECT_EQ(graph->GetDirectNode().size(), 19);
//  std::map<string, string> session_options;
//  DUMP_GRAPH_WHEN("PreRunAfterBuild");
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//  CHECK_GRAPH(PreRunAfterBuild) {
//    for (auto &node : graph->GetAllNodes()) {
//      if (node->GetName() == "conv01_0_quant_layer") {
//        bool sqrt_mode = false;
//        (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "sqrt_mode", sqrt_mode);
//        EXPECT_EQ(sqrt_mode, true);
//      }
//      if (node->GetName() == "conv02_0_quant_layer") {
//        bool sqrt_mode = false;
//        (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "sqrt_mode", sqrt_mode);
//        EXPECT_EQ(sqrt_mode, false);
//      }
//    }
//  };
//}
//
//TEST_F(STestFeWholeProcess310B, test_tbe_conv2d_smoke) {
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("tbe_conv2d_smoke.txt");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//  EXPECT_EQ(graph->GetName(), "stridedread_conv2d_stridedwrite-1_32_28_28_128_32_1_1_0_0_0_0_1_1_1_1");
//  EXPECT_EQ(graph->GetDirectNode().size(), 11);
//  std::map<string, string> session_options;
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//}
//
//TEST_F(STestFeWholeProcess310B, test_ub_fusion_common_rules) {
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("ub_fusion_common_rules.txt");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//  EXPECT_EQ(graph->GetName(), "ub_fusion_common_rules2_010");
//  EXPECT_EQ(graph->GetDirectNode().size(), 33);
//  std::map<string, string> session_options;
//  DUMP_GRAPH_WHEN("PreRunAfterBuild");
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//  std::vector<std::string> pass_names;
//  std::vector<std::string> ub_pass = {"TbeMultiOutputFusionPass"};
//  CHECK_GRAPH(PreRunAfterBuild) {
//    for (auto &node : graph->GetAllNodes()) {
//      string pass_name;
//      if (ge::AttrUtils::GetStr(node->GetOpDesc(), "pass_name_ub", pass_name) &&
//          std::find(pass_names.begin(), pass_names.end(), pass_name) == pass_names.end()) {
//        //std::cout << "pass_name_ub is " << pass_name << endl;
//        pass_names.push_back(pass_name);
//      }
//    }
//    EXPECT_EQ(pass_names, ub_pass);
//  };
//}
}
