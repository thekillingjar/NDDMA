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
#include <climits>
#include <fstream>

#include <graph_utils.h>
#include <graph_utils_ex.h>

#include "register/register_custom_pass.h"
#include "register/custom_pass_context_impl.h"
#include "framework/common/debug/ge_log.h"
#include "register/custom_pass_helper.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "graph/common/share_graph.h"

namespace ge {
class UtestRegisterStreamPass : public testing::Test { 
 protected:
  void SetUp() {}
  void TearDown() {}

  static Status AssignStreamIdByTopoIdPass(const ConstGraphPtr &graph, StreamPassContext &context) {
    if (graph->GetName() == "error_graph") {
      context.SetErrorMessage("graph name is invalid");
      return FAILED;
    }

    for (const auto &node : graph->GetAllNodes()) {
      GE_ASSERT_SUCCESS(context.SetStreamId(node, context.AllocateNextStreamId()));
    }
    return SUCCESS;
  }

  static Status AssignNewStreamIdPass(const ConstGraphPtr &graph, StreamPassContext &context) {
    for (const auto &node : graph->GetAllNodes()) {
      GE_ASSERT_SUCCESS(context.SetStreamId(node, context.AllocateNextStreamId()));
    }
    return SUCCESS;
  }
  static Status AssignStreamIdOutOfRangePass(const ConstGraphPtr &graph, StreamPassContext &context) {
    for (const auto &node : graph->GetAllNodes()) {
      if (context.SetStreamId(node, context.GetCurrMaxStreamId() + 1) != SUCCESS) {
        AscendString name;
        node.GetName(name);
        auto error_msg = AscendString("Failed to set stream id for node");
        context.SetErrorMessage(error_msg);
        return FAILED;
      }
    }
    return SUCCESS;
  }

  static Status AssignStreamIdOutOfRange2Pass(const ConstGraphPtr &graph, StreamPassContext &context) {
    for (const auto &node : graph->GetAllNodes()) {
      if (context.SetStreamId(node, -1) != SUCCESS) {
        AscendString name;
        node.GetName(name);
        auto error_msg = AscendString("Failed to set stream id for node");
        context.SetErrorMessage(error_msg);
        return FAILED;
      }
    }
    return SUCCESS;
  }

  static Status FooNonConstGraphCustomPass(GraphPtr &graph, CustomPassContext &context) {
    return SUCCESS;
  }
};

TEST_F(UtestRegisterStreamPass, AsssignStreamIdByTopoIdPass_SUCCESS) {
  int64_t default_stream_id = 0u;
  PassRegistrationData pass_reg_data("custom_pass");
  pass_reg_data.CustomAllocateStreamPassFn(AssignStreamIdByTopoIdPass).Stage(CustomPassStage::kAfterAssignLogicStream);
  CustomPassHelper::Instance().Unload();
  CustomPassHelper::Instance().Insert(pass_reg_data);

  // prepare graph
  auto compute_graph = SharedGraph<ComputeGraphPtr, ut::GraphBuilder>::BuildGraphWithControlEdge();
  compute_graph->TopologicalSorting();
  // init stream id to 0
  for (const auto &node : compute_graph->GetAllNodes()) {
    node->GetOpDesc()->SetStreamId(default_stream_id);
  }
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto const_graph = std::make_shared<const Graph>(graph);
  auto graph_ptr = std::make_shared<Graph>(graph);
  // init stream pass context
  StreamPassContext custom_pass_context(-1);

  EXPECT_EQ(CustomPassHelper::Instance().Run(graph_ptr, custom_pass_context, CustomPassStage::kAfterAssignLogicStream), SUCCESS);
  EXPECT_EQ(pass_reg_data.GetStage(), CustomPassStage::kAfterAssignLogicStream);

  // check stream id is equal with topo id
  for (const auto &node : const_graph->GetAllNodes()) {
    int64_t stream_id = custom_pass_context.GetStreamId(node);
    auto topo_id = NodeAdapter::GNode2Node(node)->GetOpDescBarePtr()->GetId();
    EXPECT_EQ(stream_id, topo_id);
  }
}

TEST_F(UtestRegisterStreamPass, AssignNewStreamIdPass_SUCCESS) {
  int64_t default_stream_id = 0u;
  PassRegistrationData pass_reg_data("custom_pass");
  pass_reg_data.CustomAllocateStreamPassFn(AssignNewStreamIdPass).Stage(CustomPassStage::kAfterAssignLogicStream);
  CustomPassHelper::Instance().Unload();
  CustomPassHelper::Instance().Insert(pass_reg_data);

  // prepare graph
  auto compute_graph = SharedGraph<ComputeGraphPtr, ut::GraphBuilder>::BuildGraphWithControlEdge();
  compute_graph->TopologicalSorting();
  // init stream id to 0
  for (const auto &node : compute_graph->GetAllNodes()) {
    node->GetOpDesc()->SetStreamId(default_stream_id);
  }
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto const_graph = std::make_shared<const Graph>(graph);
  auto graph_ptr = std::make_shared<Graph>(graph);
  // init stream pass context
  StreamPassContext custom_pass_context(0);

  EXPECT_EQ(CustomPassHelper::Instance().Run(graph_ptr, custom_pass_context, CustomPassStage::kAfterAssignLogicStream), SUCCESS);
  EXPECT_EQ(pass_reg_data.GetStage(), CustomPassStage::kAfterAssignLogicStream);

  size_t expect_stream_id = 1;
  for (const auto &node : const_graph->GetAllNodes()) {
    EXPECT_EQ(custom_pass_context.GetStreamId(node), expect_stream_id++);
  }
  EXPECT_EQ(custom_pass_context.GetCurrMaxStreamId(), 5);
}

TEST_F(UtestRegisterStreamPass, AsssignStreamIdByTopoIdPass_PassRunFailed_Failed) {
  int64_t default_stream_id = 0u;
  PassRegistrationData pass_reg_data("custom_pass");
  pass_reg_data.CustomAllocateStreamPassFn(AssignStreamIdByTopoIdPass).Stage(CustomPassStage::kAfterAssignLogicStream);
  CustomPassHelper::Instance().Unload();
  CustomPassHelper::Instance().Insert(pass_reg_data);

  // prepare graph
  auto compute_graph = SharedGraph<ComputeGraphPtr, ut::GraphBuilder>::BuildGraphWithControlEdge();
  compute_graph->TopologicalSorting();
  compute_graph->SetName("error_graph");
  // init stream id to 0
  for (const auto &node : compute_graph->GetAllNodes()) {
    node->GetOpDesc()->SetStreamId(default_stream_id);
  }
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto const_graph = std::make_shared<const Graph>(graph);
  auto graph_ptr = std::make_shared<Graph>(graph);
  // init stream pass context
  StreamPassContext custom_pass_context(0);
  for (const auto &node : const_graph->GetAllNodes()) {
    custom_pass_context.SetStreamId(node, 0u);
  }

  EXPECT_NE(CustomPassHelper::Instance().Run(graph_ptr, custom_pass_context, CustomPassStage::kAfterAssignLogicStream), SUCCESS);
  auto error_msg = custom_pass_context.GetErrorMessage().GetString();
  EXPECT_STREQ(error_msg, "graph name is invalid");
}

TEST_F(UtestRegisterStreamPass, AssignStreamIdOutOfRangePass_PassRunFailed_Failed) {
  int64_t default_stream_id = 0u;
  PassRegistrationData pass_reg_data("custom_pass");
  pass_reg_data.CustomAllocateStreamPassFn(AssignStreamIdOutOfRangePass).Stage(CustomPassStage::kAfterAssignLogicStream);
  CustomPassHelper::Instance().Unload();
  CustomPassHelper::Instance().Insert(pass_reg_data);

  // prepare graph
  auto compute_graph = SharedGraph<ComputeGraphPtr, ut::GraphBuilder>::BuildGraphWithControlEdge();
  compute_graph->TopologicalSorting();
  // init stream id to 0
  for (const auto &node : compute_graph->GetAllNodes()) {
    node->GetOpDesc()->SetStreamId(default_stream_id);
  }
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto const_graph = std::make_shared<const Graph>(graph);
  auto graph_ptr = std::make_shared<Graph>(graph);
  // init stream pass context
  StreamPassContext custom_pass_context(0);
  for (const auto &node : const_graph->GetAllNodes()) {
    custom_pass_context.SetStreamId(node, 0u);
  }

  EXPECT_NE(CustomPassHelper::Instance().Run(graph_ptr, custom_pass_context, CustomPassStage::kAfterAssignLogicStream), SUCCESS);
  auto error_msg = custom_pass_context.GetErrorMessage().GetString();
  EXPECT_STREQ(error_msg, "Failed to set stream id for node");
}
TEST_F(UtestRegisterStreamPass, AssignStreamIdOutOfRange2Pass_PassRunFailed_Failed) {
  int64_t default_stream_id = 0u;
  PassRegistrationData pass_reg_data("custom_pass");
  pass_reg_data.CustomAllocateStreamPassFn(AssignStreamIdOutOfRange2Pass).Stage(CustomPassStage::kAfterAssignLogicStream);
  CustomPassHelper::Instance().Unload();
  CustomPassHelper::Instance().Insert(pass_reg_data);

  // prepare graph
  auto compute_graph = SharedGraph<ComputeGraphPtr, ut::GraphBuilder>::BuildGraphWithControlEdge();
  compute_graph->TopologicalSorting();
  // init stream id to 0
  for (const auto &node : compute_graph->GetAllNodes()) {
    node->GetOpDesc()->SetStreamId(default_stream_id);
  }
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto const_graph = std::make_shared<const Graph>(graph);
  auto graph_ptr = std::make_shared<Graph>(graph);
  // init stream pass context
  StreamPassContext custom_pass_context(0);
  for (const auto &node : const_graph->GetAllNodes()) {
    custom_pass_context.SetStreamId(node, 0u);
  }

  EXPECT_NE(CustomPassHelper::Instance().Run(graph_ptr, custom_pass_context, CustomPassStage::kAfterAssignLogicStream), SUCCESS);
  auto error_msg = custom_pass_context.GetErrorMessage().GetString();
  EXPECT_STREQ(error_msg, "Failed to set stream id for node");
}
TEST_F(UtestRegisterStreamPass, RegNormalGraphPass_RegFailed_Failed) {
  PassRegistrationData pass_reg_data("custom_pass");
  pass_reg_data.CustomPassFn(FooNonConstGraphCustomPass).Stage(CustomPassStage::kAfterAssignLogicStream);
  CustomPassHelper::Instance().Unload();
  CustomPassHelper::Instance().Insert(pass_reg_data);

  // prepare graph
  auto compute_graph = SharedGraph<ComputeGraphPtr, ut::GraphBuilder>::BuildGraphWithControlEdge();
  compute_graph->TopologicalSorting();

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto const_graph = std::make_shared<const Graph>(graph);
  auto graph_ptr = std::make_shared<Graph>(graph);
  // init stream pass context
  StreamPassContext custom_pass_context(0);
  for (const auto &node : const_graph->GetAllNodes()) {
    custom_pass_context.SetStreamId(node, 0u);
  }

  EXPECT_NE(CustomPassHelper::Instance().Run(graph_ptr, custom_pass_context, CustomPassStage::kAfterAssignLogicStream), SUCCESS);
}
TEST_F(UtestRegisterStreamPass, StreamPassContext_ImplNull_GetStreamId_failed) {
  auto compute_graph = SharedGraph<ComputeGraphPtr, ut::GraphBuilder>::BuildGraphWithControlEdge();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto const_graph = std::make_shared<const Graph>(graph);
  auto graph_ptr = std::make_shared<Graph>(graph);

  // init stream pass context
  StreamPassContext custom_pass_context(0);
  custom_pass_context.impl_ = nullptr;

  for (const auto &node : const_graph->GetAllNodes()) {
    EXPECT_EQ(custom_pass_context.GetStreamId(node), INVALID_STREAM_ID);
  }
}
} // namespace ge
