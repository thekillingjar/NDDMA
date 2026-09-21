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
#include "ge/ge_api.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph/ge_local_context.h"
#include "graph/passes/memory_optimize/swap_space_pass.h"
#include "graph/utils/graph_utils_ex.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/build/memory/mem_assigner.h"
#include "graph/build/model_builder.h"
#include "graph/build/graph_builder.h"
#include "graph/manager/graph_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;
using namespace ge;
class TestSwapSpacePass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {
    ge::GetThreadLocalContext().SetGraphOption({});
  }
};
namespace {
const std::string OPTION_SWAP_SPACE_NODES = "ge.swapSpaceNodes";
/*
 * before swap, graph like:

            +-------------------------------------------------+
            |                                                 v
+---+     +---+     +---+     +---+     +---+     +---+     +---+     +-----------+
| A | --> | B | --> | C | --> | D | --> | E | --> | F | --> | G | --> | NetOutput |
+---+     +---+     +---+     +---+     +---+     +---+     +---+     +-----------+
                      |                             ^
                      +-----------------------------+
we need blocks as follows:
 block0 : [B]; block1 : [C]; block2 : [D, F]; block3 : [E]; block4 : [G];

after swap (with options `swap_space_nodes = "F:1;G:1"`), graph is changed like:

                                                                    +----------------------+
                                                                    |                      |
                                                                    |                      |
            +----------------------+         +----------------------+---------+            |
            |                      v         |                      |         v            v
+---+     +---+     +------+     +---+     +------+     +---+     +---+     +------+     +---+     +------+     +---+     +-----------+
| A | --> | B | --> | D2H0 | ==> | C | --> | D2H1 | ==> | D | --> | E | ==> | H2D1 | --> | F | ==> | H2D0 | --> | G | --> | NetOutput |
+---+     +---+     +------+     +---+     +------+     +---+     +---+     +------+     +---+     +------+     +---+     +-----------+
                      |            |                      ^                                |         ^            ^
                      +------------+----------------------+--------------------------------+---------+            |
                                   |                      |                                |                      |
                                   |                      |                                |                      |
                                   +----------------------+                                +----------------------+

blocks are reused as follows:
block0 : [B, D, F]; block1 : [C, E]; block2 : [G]; block3 : [H2D0, H2D1];

we can see that total memory size is reduced after swap.
*/
ComputeGraphPtr MakeGraph2() {
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
    auto F = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("F");
    auto G = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("G");
    auto netoutput1 = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput1");

    CHAIN(NODE(A)->NODE(B)->NODE(C)->NODE(D)->NODE(E)->NODE(F)->NODE(G))->NODE(netoutput1);
    CHAIN(NODE(B)->EDGE(0, 1)->NODE(G));
    CHAIN(NODE(C)->EDGE(0, 1)->NODE(F));
  };
  return ToComputeGraph(g1);
}

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


std::pair<int64_t, int64_t> GetFeatureMapMemorySize(ComputeGraphPtr compute_graph) {
  int64_t host_feature_max_offset = 0;
  int64_t device_feature_max_offset = 0;
  for (const auto &node : compute_graph->GetAllNodes()) {
    const auto &opdesc = node->GetOpDesc();
    const std::vector<int64_t> v_output_offset = opdesc->GetOutputOffset();
    std::vector<int64_t> v_memory_type;
    const bool has_mem_type_attr = AttrUtils::GetListInt(opdesc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
    for (size_t i = 0; i < v_output_offset.size(); ++i) {
      int64_t size = 0;
      TensorUtils::GetSize(opdesc->GetOutputDesc(i), size);
      if (has_mem_type_attr && (v_memory_type[i] == RT_MEMORY_HOST)) {
        host_feature_max_offset = std::max(host_feature_max_offset, v_output_offset[i] + size);
      } else if (!has_mem_type_attr || v_memory_type[i] == RT_MEMORY_HBM) {
        device_feature_max_offset = std::max(device_feature_max_offset, v_output_offset[i] + size);
      } else {
      }
    }
  }
  return std::make_pair(host_feature_max_offset, device_feature_max_offset);
}
}  // namespace

TEST_F(TestSwapSpacePass, swap_success_and_memory_peak_down) {
  int64_t device_feature_map_memory_size_before_optimize = 0;
  int64_t host_feature_map_memory_size_before_optimize = 0;
  {
    const auto &graph1 = MakeGraph2();
    std::map<std::string, std::string> options;
    Session session(options);
    DUMP_GRAPH_WHEN("PreRunAfterBuild");
    session.AddGraph(1, GraphUtilsEx::CreateGraphFromComputeGraph(graph1), options);
    std::vector<InputTensorInfo> inputs;
    auto ret = session.BuildGraph(1, inputs);
    EXPECT_EQ(ret, SUCCESS);
    CHECK_GRAPH(PreRunAfterBuild) {
      auto pair = GetFeatureMapMemorySize(graph);
      host_feature_map_memory_size_before_optimize = pair.first;
      device_feature_map_memory_size_before_optimize = pair.second;
      EXPECT_EQ(host_feature_map_memory_size_before_optimize, 0);
      EXPECT_NE(device_feature_map_memory_size_before_optimize, 0);
    };
  }
  int64_t device_feature_map_memory_size_after_optimize = 0;
  int64_t host_feature_map_memory_size_after_optimize = 0;
  {
    const auto &graph2 = MakeGraph2();
    std::string swap_space_nodes("G:1;F:1"); // peak down
//    std::string swap_space_nodes("G:1;C:0"); // cycle detect,peak no change
//    std::string swap_space_nodes("F:0;F:1"); // two input
    std::map<std::string, std::string> options;
    options.emplace(OPTION_SWAP_SPACE_NODES, swap_space_nodes);
    Session session(options);
    DUMP_GRAPH_WHEN("BeforeSwapSpace", "AfterSwapSpace", "PreRunAfterBuild");
    session.AddGraph(2, GraphUtilsEx::CreateGraphFromComputeGraph(graph2), options);
    std::vector<InputTensorInfo> inputs;
    auto ret = session.BuildGraph(2, inputs);
    EXPECT_EQ(ret, SUCCESS);
    CHECK_GRAPH(BeforeSwapSpace) {
      EXPECT_EQ(graph->GetDirectNodesSize(), 8);
    };
    CHECK_GRAPH(AfterSwapSpace) {
      EXPECT_EQ(graph->GetDirectNodesSize(), 8 + 2 * 2);
    };
    CHECK_GRAPH(PreRunAfterBuild) {
      auto pair = GetFeatureMapMemorySize(graph);
      host_feature_map_memory_size_after_optimize = pair.first;
      device_feature_map_memory_size_after_optimize = pair.second;
      EXPECT_NE(host_feature_map_memory_size_after_optimize, 0);
      EXPECT_NE(device_feature_map_memory_size_after_optimize, 0);
    };
  }
  EXPECT_LT(device_feature_map_memory_size_after_optimize, device_feature_map_memory_size_before_optimize);
}

TEST_F(TestSwapSpacePass, swap_success_and_memory_peak_down2) {
  int64_t device_feature_map_memory_size_before_optimize = 0;
  int64_t host_feature_map_memory_size_before_optimize = 0;
  {
    const auto &graph1 = MakeGraph();
    std::map<std::string, std::string> options;
    Session session(options);
    DUMP_GRAPH_WHEN("PreRunAfterBuild");
    session.AddGraph(1, GraphUtilsEx::CreateGraphFromComputeGraph(graph1), options);
    std::vector<InputTensorInfo> inputs;
    auto ret = session.BuildGraph(1, inputs);
    EXPECT_EQ(ret, SUCCESS);
    CHECK_GRAPH(PreRunAfterBuild) {
      auto pair = GetFeatureMapMemorySize(graph);
      host_feature_map_memory_size_before_optimize = pair.first;
      device_feature_map_memory_size_before_optimize = pair.second;
      EXPECT_EQ(host_feature_map_memory_size_before_optimize, 0);
      EXPECT_NE(device_feature_map_memory_size_before_optimize, 0);
    };
  }
  int64_t device_feature_map_memory_size_after_optimize = 0;
  int64_t host_feature_map_memory_size_after_optimize = 0;
  {
    const auto &graph2 = MakeGraph();
    std::string swap_space_nodes("E:1");
    std::map<std::string, std::string> options;
    options.emplace(OPTION_SWAP_SPACE_NODES, swap_space_nodes);
    Session session(options);
    DUMP_GRAPH_WHEN("BeforeSwapSpace", "AfterSwapSpace", "PreRunAfterBuild");
    session.AddGraph(2, GraphUtilsEx::CreateGraphFromComputeGraph(graph2), options);
    std::vector<InputTensorInfo> inputs;
    auto ret = session.BuildGraph(2, inputs);
    EXPECT_EQ(ret, SUCCESS);
    CHECK_GRAPH(BeforeSwapSpace) {
      EXPECT_EQ(graph->GetDirectNodesSize(), 6);
    };
    CHECK_GRAPH(AfterSwapSpace) {
      EXPECT_EQ(graph->GetDirectNodesSize(), 6 + 2 * 1);
    };
    CHECK_GRAPH(PreRunAfterBuild) {
      auto pair = GetFeatureMapMemorySize(graph);
      host_feature_map_memory_size_after_optimize = pair.first;
      device_feature_map_memory_size_after_optimize = pair.second;
      EXPECT_NE(host_feature_map_memory_size_after_optimize, 0);
      EXPECT_NE(device_feature_map_memory_size_after_optimize, 0);
    };
  }
  EXPECT_LT(device_feature_map_memory_size_after_optimize, device_feature_map_memory_size_before_optimize);
}


