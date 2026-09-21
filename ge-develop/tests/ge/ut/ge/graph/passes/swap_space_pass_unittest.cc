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
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/ge_local_context.h"
#include "macro_utils/dt_public_scope.h"
#include "graph/build/memory/mem_assigner.h"
#include "graph/build/model_builder.h"
#include "graph/manager/graph_manager.h"
#include "graph/passes/memory_optimize/swap_space_pass.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;
using namespace ge;
class UtestSwapSpacePass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {
    ge::GetThreadLocalContext().SetGraphOption({});
  }
};
namespace {
const std::string OPTION_SWAP_SPACE_NODES = "ge.swapSpaceNodes";
ComputeGraphPtr MakeGraph() {
  DEF_GRAPH(g1) {
    ge::OpDescPtr A;
    A = OP_CFG(DATA)
            .Attr(ATTR_NAME_INDEX, 0)
            .InCnt(1)
            .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
            .OutCnt(1)
            .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
            .Build("A");

    auto B = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("B");
    auto C = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("C");
    auto D = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("D");
    auto E = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("E");
    auto netoutput1 = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput1");

    CHAIN(NODE(A)->NODE(B)->NODE(C)->NODE(D)->NODE(E))->NODE(netoutput1);
    CHAIN(NODE(B)->EDGE(0, 1)->NODE(E));
  };
  return ToComputeGraph(g1);
}
}  // namespace
TEST_F(UtestSwapSpacePass, swap_success_options_test) {
  const auto &graph1 = MakeGraph();
  SwapSpacePass swap_space_pass;
  std::string swap_space_nodes("B");
  std::map<std::string, std::string> options;
  options.emplace(OPTION_SWAP_SPACE_NODES, swap_space_nodes);
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_NE(swap_space_pass.Run(graph1), SUCCESS);
  options[OPTION_SWAP_SPACE_NODES] = "E:0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(swap_space_pass.Run(graph1), SUCCESS);
}

TEST_F(UtestSwapSpacePass, swap_success_success) {
  const auto &graph1 = MakeGraph();
  SwapSpacePass swap_space_pass;
  std::string swap_space_nodes("E:0");
  std::map<std::string, std::string> options;
  options.emplace(OPTION_SWAP_SPACE_NODES, swap_space_nodes);
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(swap_space_pass.Run(graph1), SUCCESS);
}

TEST_F(UtestSwapSpacePass, get_all_swap_candidates) {
  const auto &graph1 = MakeGraph();
  SwapSpacePass swap_space_pass;
  std::string swap_space_nodes("E:0");
  std::map<std::string, std::string> options;
  options.emplace(OPTION_SWAP_SPACE_NODES, swap_space_nodes);
  ge::GetThreadLocalContext().SetGraphOption(options);
  std::map<NodePtr, SwapSpacePass::SwapInfo> swapping_candidates;
  EXPECT_EQ(swap_space_pass.GetAllSwapCandidates(graph1, swapping_candidates), SUCCESS);
  EXPECT_FALSE(swapping_candidates.empty());

  swapping_candidates.clear();
  options[OPTION_SWAP_SPACE_NODES] = std::string("E:" + std::to_string(INT64_MAX));
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(swap_space_pass.GetAllSwapCandidates(graph1, swapping_candidates), FAILED);
  EXPECT_TRUE(swapping_candidates.empty());
}
