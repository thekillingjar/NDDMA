/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// To test the engine reassign processing of dynamic data flow ops

#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "framework/common/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/partition/optimizer/dynamic_data_flow_engine_reassign_pass.h"
#include "mmpa/mmpa_api.h"

namespace ge {
class UtestDynamicDataFlowEngineReassign : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

static ComputeGraphPtr BuildGraph() {
  int32_t max_size = 100;
  GeTensorDesc tensor_desc(GeShape(), FORMAT_ND, DT_INT32);
  GeTensorPtr const_tensor =
      std::make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t *>(&max_size), sizeof(int32_t));

  const auto const_op = OP_CFG(CONSTANT).OutCnt(1).Weight(const_tensor);
  const auto stack = OP_CFG(STACK).InCnt(1).OutCnt(1).Attr(ATTR_NAME_IS_UNKNOWN_SHAPE, true);
  const auto stack_push = OP_CFG(STACKPUSH).InCnt(2).OutCnt(1).Attr(ATTR_NAME_IS_UNKNOWN_SHAPE, true);
  const auto stack_pop = OP_CFG(STACKPOP).InCnt(1).OutCnt(1).Attr(ATTR_NAME_IS_UNKNOWN_SHAPE, true);

  DEF_GRAPH(g1) {
    CHAIN(NODE("const", const_op)->EDGE(0, 0)->NODE("stack", stack));
    CHAIN(NODE("stack", stack)->EDGE(0, 0)->NODE("identity", REFIDENTITY));

    CHAIN(NODE("identity", REFIDENTITY)->EDGE(0, 0)->NODE("stackpush", stack_push));
    CHAIN(NODE("const", const_op)->EDGE(0, 1)->NODE("stackpush", stack_push));

    CHAIN(NODE("identity", REFIDENTITY)->EDGE(0, 0)->NODE("stackpop", stack_pop));
    CHAIN(NODE("stackpop", stack_pop)->EDGE(0, 0)->NODE("add", ADD));
    CHAIN(NODE("const", const_op)->EDGE(0, 1)->NODE("add", ADD));
  };

  const auto graph = ToGeGraph(g1);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void)compute_graph->TopologicalSorting();
  return compute_graph;
}

TEST_F(UtestDynamicDataFlowEngineReassign, reassign_succ) {
  char runtime2_env[MMPA_MAX_PATH] = {'0'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));

  const auto graph = BuildGraph();
  EnginePlacer placer(graph);
  const Status ret = placer.ReAssignEngine();
  EXPECT_EQ(ret, SUCCESS);

  const auto stack = graph->FindNode("stack");
  ASSERT_NE(stack, nullptr);
  const auto stack_push = graph->FindNode("stackpush");
  ASSERT_NE(stack_push, nullptr);
  const auto stack_pop = graph->FindNode("stackpop");
  ASSERT_NE(stack_pop, nullptr);
  const auto identity = graph->FindNode("identity");
  ASSERT_NE(identity, nullptr);

  static const char_t *const kGeLocalEngineName = "DNN_VM_GE_LOCAL";
  static const char_t *const kGeLocalOpKernelLibName = "DNN_VM_GE_LOCAL_OP_STORE";

  EXPECT_EQ(stack->GetOpDesc()->GetOpEngineName(), kGeLocalEngineName);
  EXPECT_EQ(stack->GetOpDesc()->GetOpKernelLibName(), kGeLocalOpKernelLibName);
  EXPECT_EQ(stack_push->GetOpDesc()->GetOpEngineName(), kGeLocalEngineName);
  EXPECT_EQ(stack_push->GetOpDesc()->GetOpKernelLibName(), kGeLocalOpKernelLibName);
  EXPECT_EQ(stack_pop->GetOpDesc()->GetOpEngineName(), kGeLocalEngineName);
  EXPECT_EQ(stack_pop->GetOpDesc()->GetOpKernelLibName(), kGeLocalOpKernelLibName);
  EXPECT_EQ(identity->GetOpDesc()->GetOpEngineName(), "");
  EXPECT_EQ(identity->GetOpDesc()->GetOpKernelLibName(), "");

  const NodeEngineMap &node_atomic_engine_map = placer.GetNodeEngineMap(false);
  EXPECT_EQ(node_atomic_engine_map.size(), 3);
  EXPECT_EQ(node_atomic_engine_map.at(stack), kGeLocalEngineName);
  EXPECT_EQ(node_atomic_engine_map.at(stack_push), kGeLocalEngineName);
  EXPECT_EQ(node_atomic_engine_map.at(stack_pop), kGeLocalEngineName);
}

TEST_F(UtestDynamicDataFlowEngineReassign, rtv2_skip_reassign) {
  char runtime2_env[MMPA_MAX_PATH] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));

  const auto graph = BuildGraph();
  EnginePlacer placer(graph);
  const Status ret = placer.ReAssignEngine();
  EXPECT_EQ(ret, SUCCESS);

  const auto stack = graph->FindNode("stack");
  ASSERT_NE(stack, nullptr);
  const auto stack_push = graph->FindNode("stackpush");
  ASSERT_NE(stack_push, nullptr);
  const auto stack_pop = graph->FindNode("stackpop");
  ASSERT_NE(stack_pop, nullptr);
  const auto identity = graph->FindNode("identity");
  ASSERT_NE(identity, nullptr);

  static const char_t *const kGeLocalEngineName = "DNN_VM_GE_LOCAL";
  static const char_t *const kGeLocalOpKernelLibName = "DNN_VM_GE_LOCAL_OP_STORE";

  EXPECT_NE(stack->GetOpDesc()->GetOpEngineName(), kGeLocalEngineName);
  EXPECT_NE(stack->GetOpDesc()->GetOpKernelLibName(), kGeLocalOpKernelLibName);
  EXPECT_NE(stack_push->GetOpDesc()->GetOpEngineName(), kGeLocalEngineName);
  EXPECT_NE(stack_push->GetOpDesc()->GetOpKernelLibName(), kGeLocalOpKernelLibName);
  EXPECT_NE(stack_pop->GetOpDesc()->GetOpEngineName(), kGeLocalEngineName);
  EXPECT_NE(stack_pop->GetOpDesc()->GetOpKernelLibName(), kGeLocalOpKernelLibName);
  EXPECT_EQ(identity->GetOpDesc()->GetOpEngineName(), "");
  EXPECT_EQ(identity->GetOpDesc()->GetOpKernelLibName(), "");

  const NodeEngineMap &node_atomic_engine_map = placer.GetNodeEngineMap(false);
  EXPECT_TRUE(node_atomic_engine_map.empty());
}
}  // namespace ge
