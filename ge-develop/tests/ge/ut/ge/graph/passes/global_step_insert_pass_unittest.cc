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

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/feature/global_step_insert_pass.h"

#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/passes/base_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/tuning_utils.h"
#include "graph_builder_utils.h"
#include "graph/ge_context.h"
#include "graph/passes/pass_manager.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/ge_local_context.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestGlobalStepInsertPass : public Test {
 protected:
};

static ComputeGraphPtr BuildGraph1() {
  ge::ut::GraphBuilder builder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto identity1 = builder.AddNode("identity1", "Identity", 1, 1);
  auto out = builder.AddNode("out", "NetOutput", 1, 1);

  builder.AddDataEdge(var1, 0, identity1, 0);
  builder.AddControlEdge(var2, identity1);
  builder.AddDataEdge(identity1, 0, out, 0);
  return builder.GetGraph();
}

TEST_F(UtestGlobalStepInsertPass, skip_insert) {
  auto graph = BuildGraph1();
  GlobalStepInsertPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  NodePtr found_node = graph->FindNode(NODE_NAME_GLOBAL_STEP);
  EXPECT_NE(found_node, nullptr);
}

TEST_F(UtestGlobalStepInsertPass, InsertOp) {
  auto graph = BuildGraph1();
  GlobalStepInsertPass pass;
  std::vector<GeTensorDesc> input_list;
  std::vector<GeTensorDesc> output_list;

  vector<int64_t> dim_h({1, 2, 3, 4});
  GeShape shape_h(dim_h);
  GeTensorDesc tensor_desc_h(shape_h);
  tensor_desc_h.SetOriginFormat(FORMAT_NHWC);
  tensor_desc_h.SetFormat(FORMAT_ND);
  tensor_desc_h.SetDataType(DT_FLOAT);
  input_list.push_back(tensor_desc_h);
  output_list.push_back(tensor_desc_h);
 
  NodePtr ptr = pass.InsertOp(graph, "node_type", "node_name", input_list, output_list);
  // EXPECT_EQ(status, SUCCESS);

  
  GeTensorDesc tensor_desc;
  input_list.push_back(tensor_desc);
  output_list.push_back(tensor_desc);
  ptr = pass.InsertOp(graph, "type", "name", input_list, output_list);
  EXPECT_NE(ptr, nullptr);
}

TEST_F(UtestGlobalStepInsertPass, RunSucess) {
  auto graph = std::make_shared<ge::ComputeGraph>("test_graph");
  GlobalStepInsertPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(UtestGlobalStepInsertPass, test_no_need_to_insert) {
  const auto graph = std::make_shared<ge::ComputeGraph>("test_graph");
  GlobalStepInsertPass pass;
  // 1. Host exec
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options["ge.exec.placement"] = "HOST";
  GetThreadLocalContext().SetGraphOption(graph_options);
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  graph_options.erase("ge.exec.placement");
  GetThreadLocalContext().SetGraphOption(graph_options);

  // 2. Already has global step
  ge::ut::GraphBuilder builder("g1");
  auto global_step = builder.AddNode("ge_global_step", "Variable", 0, 1);
  auto identity1 = builder.AddNode("identity1", "Identity", 1, 1);
  auto out = builder.AddNode("out", "NetOutput", 1, 1);
  builder.AddDataEdge(global_step, 0, identity1, 0);
  builder.AddDataEdge(identity1, 0, out, 0);
  const auto graph2 = builder.GetGraph();
  status = pass.Run(graph2);
  EXPECT_EQ(status, SUCCESS);

  // 3. subgraph
  graph2->SetParentGraph(graph);
  status = pass.Run(graph2);
  EXPECT_EQ(status, SUCCESS);
}
