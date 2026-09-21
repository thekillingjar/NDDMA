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
#include <fstream>
#include <sstream>
#include <algorithm>
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

using namespace std;

namespace fe {
class STestFeWholeProcessNano: public testing::Test
{
protected:
  static void SetUpTestCase() {
    const char *soc_version = "Ascend035";
    rtSetSocVersion(soc_version);
    auto &fe_env = fe_env::FeRunningEnv::Instance();
    std::map<string, string> new_options;
    new_options.emplace(ge::SOC_VERSION, soc_version);
    new_options.emplace(ge::AICORE_NUM, "1");
    new_options.emplace(ge::CORE_TYPE, AI_CORE_TYPE);
    new_options.emplace(ge::PRECISION_MODE, FORCE_FP16);
    fe_env.InitRuningEnv(new_options);
  }

  static void TearDownTestCase() {
    te::TeStub::GetInstance().SetImpl(std::make_shared<te::TeStubApi>());
    auto &fe_env = fe_env::FeRunningEnv::Instance();
    fe_env.FinalizeEnv();
  }
};

//TEST_F(STestFeWholeProcessNano, test_ub_fusion_cube_quant_net) {
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("nano_quant");
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("nano_quant_begin.txt");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//  EXPECT_EQ(graph->GetName(), "quantized");
//  EXPECT_EQ(graph->GetDirectNode().size(), 123);
//  DUMP_GRAPH_WHEN("PreRunAfterBuild");
//  std::map<string, string> session_options;
//  session_options.emplace(ge::OUTPUT_DATATYPE, "FP16");
//  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
//  EXPECT_EQ(graph->GetDirectNode().size(), 44);
//  CHECK_GRAPH(PreRunAfterBuild) {
//    EXPECT_EQ(graph->GetDirectNode().size(), 40);
//  };
//}
}
