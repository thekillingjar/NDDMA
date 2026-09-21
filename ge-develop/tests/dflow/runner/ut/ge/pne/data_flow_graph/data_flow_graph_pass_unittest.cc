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

#include "dflow/compiler/data_flow_graph/convert_batch_attr_to_udf_pass.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph_prune_pass.h"
#include "flow_graph/data_flow.h"
#include "graph/utils/graph_utils_ex.h"

using namespace testing;

namespace ge {
class DataFlowGraphPassTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(DataFlowGraphPassTest, TimeBatch_CountBatch_Run_Success) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("Node0", 2, 1).SetInput(0, data0).SetInput(1, data1);
  auto function_pp = dflow::FunctionPp("function_pp");
  node0.AddPp(function_pp);
  dflow::TimeBatch time_batch = {0};
  time_batch.time_window = -1;
  dflow::DataFlowInputAttr input0_attr = {dflow::DataFlowAttrType::TIME_BATCH, &time_batch};
  node0.MapInput(0, function_pp, 0, {input0_attr});

  dflow::CountBatch count_batch = {0};
  count_batch.batch_size = 1;
  dflow::DataFlowInputAttr input1_attr = {dflow::DataFlowAttrType::COUNT_BATCH, &count_batch};
  node0.MapInput(1, function_pp, 1, {input1_attr});

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph("FlowGraph");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  auto graph = GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());

  ConvertBatchAttrToUdfPass pass;
  ASSERT_EQ(graph->GetAllNodes().size(), 4);
  ASSERT_EQ(pass.Run(graph), SUCCESS);
  ASSERT_EQ(graph->GetAllNodes().size(), 6);
  auto node_node0_time_batch = graph->FindNode("Node0__BuiltIn_TimeBatch_0");
  ASSERT_NE(node_node0_time_batch, nullptr);
  auto node_data0 = graph->FindNode("Data0");
  auto node_node0 = graph->FindNode("Node0");
  ASSERT_EQ(node_data0->GetOutDataAnchor(0)->IsLinkedWith(node_node0_time_batch->GetInDataAnchor(0)), true);
  ASSERT_EQ(node_node0_time_batch->GetOutDataAnchor(0)->IsLinkedWith(node_node0->GetInDataAnchor(0)), true);
  auto node_node0_count_batch = graph->FindNode("Node0__BuiltIn_CountBatch_1");
  EXPECT_NE(node_node0_count_batch, nullptr);
  auto node_data1 = graph->FindNode("Data1");
  EXPECT_EQ(node_data1->GetOutDataAnchor(0)->IsLinkedWith(node_node0_count_batch->GetInDataAnchor(0)), true);
  EXPECT_EQ(node_node0_count_batch->GetOutDataAnchor(0)->IsLinkedWith(node_node0->GetInDataAnchor(1)), true);
}

TEST_F(DataFlowGraphPassTest, TimeBatch_CountBatch_Run_With_Catch_Exception_Failed) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("Node0", 2, 1).SetInput(0, data0).SetInput(1, data1);
  auto function_pp = dflow::FunctionPp("function_pp");
  node0.AddPp(function_pp);
  dflow::TimeBatch time_batch = {0};
  time_batch.time_window = -1;
  dflow::DataFlowInputAttr input0_attr = {dflow::DataFlowAttrType::TIME_BATCH, &time_batch};
  node0.MapInput(0, function_pp, 0, {input0_attr});

  dflow::CountBatch count_batch = {0};
  count_batch.batch_size = 1;
  dflow::DataFlowInputAttr input1_attr = {dflow::DataFlowAttrType::COUNT_BATCH, &count_batch};
  node0.MapInput(1, function_pp, 1, {input1_attr});

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph("FlowGraph");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  auto graph = GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());
  AttrUtils::SetBool(graph, "_enable_exception_catch", true);

  ConvertBatchAttrToUdfPass pass;
  ASSERT_EQ(graph->GetAllNodes().size(), 4);
  ASSERT_EQ(pass.Run(graph), FAILED);
}

TEST_F(DataFlowGraphPassTest, TimeBatch_With_Invalid_Window) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("Node0", 2, 1).SetInput(0, data0).SetInput(1, data1);
  auto function_pp = dflow::FunctionPp("function_pp");
  node0.AddPp(function_pp);
  dflow::TimeBatch time_batch = {0};
  time_batch.time_window = 0;
  dflow::DataFlowInputAttr input0_attr = {dflow::DataFlowAttrType::TIME_BATCH, &time_batch};
  node0.MapInput(0, function_pp, 0, {input0_attr});
  node0.MapInput(1, function_pp, 1);

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph("FlowGraph");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  auto graph = GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());

  ConvertBatchAttrToUdfPass pass;
  ASSERT_EQ(graph->GetAllNodes().size(), 4);
  ASSERT_NE(pass.Run(graph), SUCCESS);
}

TEST_F(DataFlowGraphPassTest, Prune_pass_succ) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto data2 = dflow::FlowData("Data2", 2);
  auto node0 = dflow::FlowNode("Node0", 2, 1).SetInput(0, data0).SetInput(1, data1);
  auto node1 = dflow::FlowNode("Node1", 2, 1).SetInput(0, data0).SetInput(1, data2);

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1, data2};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph("FlowGraph");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  auto graph = GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());

  DataFlowGraphPrunePass pass;
  ASSERT_EQ(pass.Run(nullptr), GE_GRAPH_ISNULL);
  ASSERT_EQ(graph->GetDirectNode().size(), 6);
  ASSERT_EQ(pass.Run(graph), SUCCESS);
  ASSERT_EQ(graph->GetDirectNode().size(), 5);
}

TEST_F(DataFlowGraphPassTest, Prune_pass_without_outnodes) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto data2 = dflow::FlowData("Data2", 2);
  auto node0 = dflow::FlowNode("Node0", 2, 1).SetInput(0, data0).SetInput(1, data1);
  auto node1 = dflow::FlowNode("Node1", 2, 1).SetInput(0, data0).SetInput(1, data2);

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1, data2};

  dflow::FlowGraph flow_graph("FlowGraph");
  flow_graph.SetInputs(inputsOperator);
  auto graph = GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());

  DataFlowGraphPrunePass pass;
  ASSERT_EQ(graph->GetDirectNode().size(), 5);
  ASSERT_EQ(pass.Run(graph), SUCCESS);
  ASSERT_EQ(graph->GetDirectNode().size(), 5);
}

TEST_F(DataFlowGraphPassTest, Prune_pass_with_send_succ) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto data2 = dflow::FlowData("Data2", 2);
  auto node0 = dflow::FlowNode("Node0", 2, 1).SetInput(0, data0).SetInput(1, data1);
  auto node1 = dflow::FlowNode("Node1", 2, 1).SetInput(0, data0).SetInput(1, data2);

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1, data2};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph("FlowGraph");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  auto graph = GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());

  DataFlowGraphPrunePass pass;
  ASSERT_EQ(pass.Run(nullptr), GE_GRAPH_ISNULL);
  ASSERT_EQ(graph->GetDirectNode().size(), 6);
  ASSERT_EQ(pass.Run(graph), SUCCESS);
  ASSERT_EQ(graph->GetDirectNode().size(), 5);
}
}  // namespace ge
