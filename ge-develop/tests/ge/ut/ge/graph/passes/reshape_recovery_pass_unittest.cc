/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/shape_optimize/reshape_recovery_pass.h"

#include <gtest/gtest.h>
#include <set>
#include <string>
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "graph/passes/pass_manager.h"
#include "graph_builder_utils.h"

namespace ge {
class UtestReshapeRecoveryPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
const std::string kVar1 = "var1";
const std::string kVar2 = "var2";
const std::string kTransdata1 = "transdata1";
const std::string kTransdata2 = "transdata2";
const std::string kNetoutput = "netoutput1";

/*
 *          Netoutput       +-----------+  +-----------+
 *             |            |Then Graph |  |Else Graph |
 *             |            |NetOutput2 |  |NetOutput2 |
 *         transdata2       |     |     |  |     |     |
 *            |             |transdata4 |  |transdata4 |
 *         transdata1       |    |      |  |    |      |
 *           |              |transdata3 |  |transdata3 |
 *          if0 <---------> |   |       |  |    |      |
 *        /    \            | Data(1)   |  | Data(1)   |
 * pred(Data)  input(Data)  +-----------+  +-----------+
 */
ComputeGraphPtr BuildIfGraph() {
  auto main_graph = []() {
    ut::GraphBuilder builder = ut::GraphBuilder("main_builder");
    auto pred = builder.AddNode("pred", "Data", 0, 1, FORMAT_ND, DT_FLOAT, {1, 256, 256});
    auto input = builder.AddNode("input", "Data", 0, 1, FORMAT_ND, DT_FLOAT, {1, 256, 256});
    auto if0 = builder.AddNode("if0", "If", 2, 1, FORMAT_ND, DT_FLOAT, {256, 256});
    auto transdata1 = builder.AddNode("transdata1", "Transdata", 1, 1, FORMAT_ND, DT_FLOAT, {1, 256, 256});
    auto transdata2 = builder.AddNode("transdata2", "Transdata", 1, 1, FORMAT_ND, DT_FLOAT, {256, 256});
    auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0, FORMAT_ND, DT_FLOAT, {1, 256, 256});

    builder.AddDataEdge(pred, 0, if0, 0);
    builder.AddDataEdge(input, 0, if0, 1);
    builder.AddDataEdge(if0, 0, transdata1, 0);
    builder.AddDataEdge(transdata1, 0, transdata2, 0);
    builder.AddDataEdge(transdata2, 0, netoutput1, 0);
    return builder.GetGraph();
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("pred")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("input")->GetOpDesc(), "index", 1);

  auto sub_graph_builder = [](const std::string &builder_name) {
    ut::GraphBuilder builder = ut::GraphBuilder(builder_name);
    auto data = builder.AddNode("data", "Data", 0, 1, FORMAT_ND, DT_FLOAT, {1, 256, 256});
    auto transdata3 = builder.AddNode("transdata3", "Transdata", 1, 1, FORMAT_ND, DT_FLOAT, {256, 256});
    auto transdata4 = builder.AddNode("transdata4", "Transdata", 1, 1, FORMAT_ND, DT_FLOAT, {1, 256, 256});
    auto netoutput2 = builder.AddNode("netoutput2", "NetOutput", 1, 0, FORMAT_ND, DT_FLOAT, {256, 256});
    builder.AddDataEdge(data, 0, transdata3, 0);
    builder.AddDataEdge(transdata3, 0, transdata4, 0);
    builder.AddDataEdge(transdata4, 0, netoutput2, 0);
    return builder;
  };

  auto then_graph = sub_graph_builder("then_builder").GetGraph();
  then_graph->SetName("then");
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(then_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto else_graph = sub_graph_builder("else_builder").GetGraph();
  else_graph->SetName("else");
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(else_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  then_graph->SetParentGraph(main_graph);
  then_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(main_graph);
  else_graph->SetParentNode(if_node);

  main_graph->AddSubgraph(then_graph);
  main_graph->AddSubgraph(else_graph);
  if_node->GetOpDesc()->AddSubgraphName("then");
  if_node->GetOpDesc()->AddSubgraphName("else");
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then");
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else");
  if_node->GetOpDesc()->AppendIrInput("cond", kIrInputRequired);
  if_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);
  main_graph->TopologicalSorting();

  return main_graph;
}

/**
 *    netoutput1
 *     |       \.
 *transdata1    \.
 *    |          \.
 *    |   transdata2
 *    |        /
 *   var1   var2
 */
ut::GraphBuilder GraphBuilderByGivenShapes(const map<string, std::vector<int64_t>> node_name_2_shapes) {
  ut::GraphBuilder builder = ut::GraphBuilder("g2");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1, FORMAT_ND, DT_FLOAT, node_name_2_shapes.at(kVar1));
  auto var2 = builder.AddNode("var2", "Variable", 0, 1, FORMAT_ND, DT_FLOAT, node_name_2_shapes.at(kVar2));
  auto transdata1 =
      builder.AddNode("transdata1", "Transdata", 1, 1, FORMAT_ND, DT_FLOAT, node_name_2_shapes.at(kTransdata1));
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 2, 0);
  auto transdata2 =
      builder.AddNode("transdata2", "Transdata", 1, 1, FORMAT_ND, DT_FLOAT, node_name_2_shapes.at(kTransdata2));

  builder.AddDataEdge(var1, 0, transdata1, 0);
  builder.AddDataEdge(var2, 0, transdata2, 0);
  builder.AddDataEdge(transdata1, 0, netoutput1, 0);
  builder.AddDataEdge(transdata2, 0, netoutput1, 1);
  return builder;
}

ut::GraphBuilder Graph1Builder() {
  const std::map<std::string, std::vector<int64_t>> node_name_2_shapes = {
      {kVar1, {-1}},
      {kVar2, {1, 1, 224, 224}},
      {kTransdata1, {-1, 224}},
      {kTransdata2, {224, 224}}
  };
  return GraphBuilderByGivenShapes(node_name_2_shapes);
}

void TestReShapeRecoveryPass(const std::map<std::string, std::vector<int64_t>> &node_name_2_shapes,
                             const std::map<std::string, size_t> &node_type_2_node_counts,
                             ComputeGraphPtr &compute_graph_after_pass) {
  auto builder = GraphBuilderByGivenShapes(node_name_2_shapes);
  auto graph = builder.GetGraph();
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5U);
  ReshapeRecoveryPass reshape_recovery_pass;
  Status ret = reshape_recovery_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  compute_graph_after_pass = graph;
  EXPECT_EQ(gert::SummaryChecker(graph).StrictDirectNodeTypes(node_type_2_node_counts), "success");
}
}  // namespace

TEST_F(UtestReshapeRecoveryPass, reshape_recovery_with_dynamic_shape) {
  auto builder = Graph1Builder();
  auto graph = builder.GetGraph();
  ReshapeRecoveryPass reshape_recovery_pass;
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  Status ret = reshape_recovery_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  // 5+ reshape *2 + shape_const *2 = 9
  EXPECT_EQ(graph->GetDirectNodesSize(), 9);

  auto reshape1 = graph->FindNode("Reshape_ReshapeRecoveryPass_0");
  EXPECT_NE(reshape1, nullptr);
}

// 预置条件：纯动态shape节点
// 预期结果：拓扑不改变，只刷新Netoutput节点的输入shape
TEST_F(UtestReshapeRecoveryPass, AllDynamicNodes_TopoNotChange) {
  const std::map<std::string, std::vector<int64_t>> node_name_2_shapes = {
      {kVar1, {-1}},
      {kVar2, {-1, 1, 224, 224}},
      {kTransdata1, {-1, 224}},
      {kTransdata2, {-224, 224}}
  };
  const std::map<std::string, size_t> node_type_2_node_counts = {{"Variable", 2}, {"Transdata", 2}, {"NetOutput", 1}};

  ComputeGraphPtr compute_graph_after_pass;
  TestReShapeRecoveryPass(node_name_2_shapes, node_type_2_node_counts, compute_graph_after_pass);
  const auto netoutput_node = compute_graph_after_pass->FindNode(kNetoutput);
  ASSERT_NE(netoutput_node, nullptr);
  EXPECT_EQ(netoutput_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims(),
            std::vector<int64_t>(node_name_2_shapes.at(kTransdata1)));
  EXPECT_EQ(netoutput_node->GetOpDesc()->GetInputDesc(1).GetShape().GetDims(),
            std::vector<int64_t>(node_name_2_shapes.at(kTransdata2)));
}

// 预置条件：纯静态shape节点,各节点shape均不同
// 预期结果：4Reshape 3const
TEST_F(UtestReshapeRecoveryPass, AllStaticNode_DstShapeIsDifferent) {
  const std::map<std::string, std::vector<int64_t>> node_name_2_shapes = {
      {kVar1, {224, 224}},
      {kVar2, {1, 1, 224, 224}},
      {kTransdata1, {1, 224, 224}},
      {kTransdata2, {224, 224}}
  };
  const std::map<std::string, size_t> node_type_2_node_counts = {
      {"Reshape", 4}, {"Const", 3}, {"Variable", 2}, {"Transdata", 2}, {"NetOutput", 1}};

  ComputeGraphPtr compute_graph_after_pass;
  TestReShapeRecoveryPass(node_name_2_shapes, node_type_2_node_counts, compute_graph_after_pass);
}

// 预置条件：纯静态shape节点,两条variable输入分支的shape均相同
// 预期结果：4Reshape 2const
TEST_F(UtestReshapeRecoveryPass, AllStaticNode_DstShapeIsTheSame) {
  const std::map<std::string, std::vector<int64_t>> node_name_2_shapes = {
      {kVar1, {1, 1, 224, 224}},
      {kVar2, {1, 1, 224, 224}},
      {kTransdata1, {224, 224}},
      {kTransdata2, {224, 224}}
  };
  const std::map<std::string, size_t> node_type_2_node_counts = {
      {"Reshape", 4}, {"Const", 2}, {"Variable", 2}, {"Transdata", 2}, {"NetOutput", 1}};

  ComputeGraphPtr compute_graph_after_pass;
  TestReShapeRecoveryPass(node_name_2_shapes, node_type_2_node_counts, compute_graph_after_pass);
  for (const auto &node : compute_graph_after_pass->GetDirectNode()) {
    if (node->GetType() == "Const") {
      EXPECT_EQ(gert::NodeTopoChecker(node).StrictConnectTo(0, {{"Reshape"}, {"Reshape"}}), "success");
    }
  }
}

// 操作步骤：pass执行两次
// 预期结果：OM一致
TEST_F(UtestReshapeRecoveryPass, AllStaticNode_PassRun2Times_OMIsTheSame) {
  const std::map<std::string, std::vector<int64_t>> node_name_2_shapes = {
      {kVar1, {1, 1, 224, 224}},
      {kVar2, {1, 1, 224, 224}},
      {kTransdata1, {224, 224}},
      {kTransdata2, {224, 224}}
  };
  const std::map<std::string, size_t> node_type_2_node_counts = {
      {"Reshape", 4}, {"Const", 2}, {"Variable", 2}, {"Transdata", 2}, {"NetOutput", 1}};

  ComputeGraphPtr compute_graph_after_pass;
  TestReShapeRecoveryPass(node_name_2_shapes, node_type_2_node_counts, compute_graph_after_pass);
  TestReShapeRecoveryPass(node_name_2_shapes, node_type_2_node_counts, compute_graph_after_pass);
}

// 用例描述：子图场景测试
// 预置条件：If根图和子图shape均不连续，且带有相同目标shape
// 操作步骤：构建If图-->pass run
// 预期结果：各子图内const节点存在复用，子图间、子图与根图不存在const复用
TEST_F(UtestReshapeRecoveryPass, IfSubgraphWithAllStaticNodes_CheckResultTopoOk) {
  auto if_graph = BuildIfGraph();
  ASSERT_NE(if_graph, nullptr);
  EXPECT_EQ(if_graph->GetDirectNodesSize(), 6U);
  EXPECT_EQ(if_graph->GetAllNodesSize(), 6U + 4U * 2);

  PassManager graph_pass;
  graph_pass.AddPass("ReshapeRecoveryPass", new (std::nothrow) ReshapeRecoveryPass);
  Status ret = graph_pass.Run(if_graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(gert::SummaryChecker(if_graph).StrictDirectNodeTypes(
                {{"Data", 2}, {"Transdata", 2}, {"If", 1}, {"Reshape", 5}, {"Const", 2}, {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(if_graph->GetAllSubgraphs().size(), 2U);
  for (const auto &sub_graph : if_graph->GetAllSubgraphs()) {
    EXPECT_EQ(gert::SummaryChecker(sub_graph).StrictDirectNodeTypes(
                  {{"Data", 1}, {"Transdata", 2}, {"Reshape", 3}, {"Const", 2}, {"NetOutput", 1}}),
              "success");
  }
}
}  // namespace ge
