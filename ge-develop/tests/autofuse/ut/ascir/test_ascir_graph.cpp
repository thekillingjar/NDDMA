/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "ascir.h"
#include "ascir_ops.h"

#include "node_utils_ex.h"
#include "graph_utils_ex.h"
using namespace ascir;

TEST(Ascir_Graph, AddStartNode_Ok) {
  Graph g("graph");
  Data data("data");
  g.AddNode(data);

  auto cg = ge::GraphUtilsEx::GetComputeGraph(g);
  ASSERT_NE(cg, nullptr);
  auto data_node = cg->FindNode("data");
  ASSERT_NE(data_node, nullptr);
  auto data_node_in_op = ge::NodeUtilsEx::GetNodeFromOperator(data);
  ASSERT_NE(data_node_in_op, nullptr);
  ASSERT_EQ(data_node, data_node_in_op);
}

TEST(Ascir_Graph_Bg, AddStartNode_Ok) {
  Graph g("graph");
  auto data = cg::Data("data", g);

  auto cg = ge::GraphUtilsEx::GetComputeGraph(g);
  ASSERT_NE(cg, nullptr);
  auto data_node = cg->FindNode("data");
  ASSERT_NE(data_node, nullptr);
  auto data_node_in_op = ge::NodeUtilsEx::GetNodeFromOperator(data);
  ASSERT_NE(data_node_in_op, nullptr);
  ASSERT_EQ(data_node, data_node_in_op);
}

static std::string SizeExprStr(const ascir::Graph &graph, const ascir::SizeExpr &size_expr) {
  std::stringstream ss;

  auto var_to_name = [&graph](ascir::SizeVarId id, void*) -> string{
    return graph.size_var[id].name;
  };

  return size_expr.String(var_to_name);
}

static std::stringstream &SizeExprListStr(std::stringstream &ss, const ascir::Graph &graph,
                                          const std::vector<ascir::SizeExpr> &size_expr_list) {
  for (auto& size_expr: size_expr_list) {
    ss << SizeExprStr(graph, size_expr) << ", ";
  }
  return ss;
}

TEST(Ascir_Graph_Bg, SetAxisMap_Ok) {
  ascir::Graph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  auto data = ascir::cg::ContiguousData("test_op1", graph, ge::DT_FLOAT, {});
  auto cast = ascir::cg::Cast("cast", data.y, ge::DT_FLOAT16);
  cast.attr.sched.axis = {z0.id, z1.id};
  cast.y.axis = {z0.id, z1.id};
  cast.y.repeats = {s0, s1};
  cast.y.strides = {s1, 1};
  graph.SetInputs({data});

  std::map<AxisId, vector<AxisId>> axis_map;
  axis_map[0] = {2, 3};
  axis_map[1] = {2, 3};

  graph.SetAxisMap(axis_map);
  std::stringstream axis_ss;
  for (auto axis_id : cast.attr.sched.axis()) {
    axis_ss << graph.axis[axis_id].name << ", ";
  }
  EXPECT_EQ(axis_ss.str(), "z2, z3, z2, z3, ");
  std::stringstream repeat_ss;
  SizeExprListStr(repeat_ss, graph, cast.y.repeats);
  EXPECT_EQ(repeat_ss.str(), "s2, s3, s2, s3, ");
  std::stringstream stride_ss;
  SizeExprListStr(stride_ss, graph, cast.y.strides);
  EXPECT_EQ(stride_ss.str(), "s1 * s3, s1, s3, 1, ");
}