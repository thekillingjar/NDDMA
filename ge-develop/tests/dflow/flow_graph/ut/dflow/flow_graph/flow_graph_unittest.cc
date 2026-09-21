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
#include "flow_graph/data_flow.h"
#include "proto/dflow.pb.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "graph/operator_reg.h"
#include "graph/operator_factory_impl.h"

using namespace ge::dflow;

namespace ge {
class FlowGraphUTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    is_data_exits_ = OperatorFactoryImpl::IsExistOp("Data");
    if (!is_data_exits_) {
      OperatorFactoryImpl::RegisterOperatorCreator("Data", [](const std::string &name) {
        auto op_desc = std::make_shared<OpDesc>(name, "Data");
        return OpDescUtils::CreateOperatorFromOpDesc(op_desc);
      });
    }
    is_flow_node_exits_ = OperatorFactoryImpl::IsExistOp("FlowNode");
    if (!is_flow_node_exits_) {
      OperatorFactoryImpl::RegisterOperatorCreator("FlowNode", [](const std::string &name) {
        auto op_desc = std::make_shared<OpDesc>(name, "FlowNode");
        return OpDescUtils::CreateOperatorFromOpDesc(op_desc);
      });
    }
  }
  static void TearDownTestCase() {
    if (!is_data_exits_) {
      (*OperatorFactoryImpl::operator_creators_).erase("Data");
    }
    if (!is_flow_node_exits_) {
      (*OperatorFactoryImpl::operator_creators_).erase("FlowNode");
    }
  }
  void SetUp() {}
  void TearDown() {}
 private:
  static bool is_data_exits_;
  static bool is_flow_node_exits_;
};
bool FlowGraphUTest::is_data_exits_ = false;
bool FlowGraphUTest::is_flow_node_exits_ = false;

TEST_F(FlowGraphUTest, DflowFuncBasicTest_AddPp) {
  auto data0 = FlowData("Data0", 0);
  auto data1 = FlowData("Data1", 1);
  auto data2 = FlowData("Data2", 2);
  ge::Graph graph("user_graph");
  GraphBuilder graph_build = [graph]() { return graph; };
  auto pp1 = GraphPp("pp1", graph_build).SetCompileConfig("./pp1.json");
  auto node0 = FlowNode("node0", 3, 2).SetInput(0, data0).SetInput(1, data1).SetInput(2, data2).AddPp(pp1).AddPp(pp1);

  std::vector<std::string> pp_attrs;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node0);
  ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs);
  auto process_point = dataflow::ProcessPoint();
  auto flag = process_point.ParseFromString(pp_attrs[0]);
  ASSERT_TRUE(flag);
  ASSERT_EQ(process_point.name(), "pp1");
  ASSERT_EQ(process_point.type(), dataflow::ProcessPoint_ProcessPointType_GRAPH);
  ASSERT_EQ(process_point.compile_cfg_file(), "./pp1.json");
  ASSERT_EQ(process_point.funcs_size(), 0);
  ASSERT_EQ(process_point.graphs_size(), 1);
  ASSERT_EQ(process_point.invoke_pps_size(), 0);
  ASSERT_EQ(process_point.in_edges_size(), 0);
  ASSERT_EQ(process_point.out_edges_size(), 0);
}

TEST_F(FlowGraphUTest, DflowFuncBasicTest_SetGraphPpBuilderAsync) {
  auto data0 = FlowData("Data0", 0);
  ge::Graph graph("user_graph");
  GraphBuilder graph_build = [graph]() { return graph; };
  auto pp1 = GraphPp("pp1", graph_build).SetCompileConfig("./pp1.json");
  auto node0 = FlowNode("node0", 1, 1);
  node0.AddPp(pp1);
  node0.SetInput(0, data0);

  FlowGraph flow_graph("flow_graph");
  FlowGraphBuilder flow_graph_build = [flow_graph]() { return flow_graph; };
  std::vector<FlowOperator> inputs_operator{data0};
  std::vector<FlowOperator> outputs_operator{node0};
  flow_graph.SetGraphPpBuilderAsync(true);
  flow_graph.SetInputs(inputs_operator).SetOutputs(outputs_operator);
  ASSERT_EQ(strcmp(flow_graph.GetName(), "flow_graph"), 0);
  ASSERT_EQ(flow_graph.ToGeGraph().GetName(), "flow_graph");
}

TEST_F(FlowGraphUTest, DflowFuncBasicTest_Map) {
  auto data0 = FlowData("Data0", 0);
  auto data1 = FlowData("Data1", 1);
  auto data2 = FlowData("Data2", 2);
  auto pp1 = FunctionPp("pp1").SetCompileConfig("./pp1.json");
  auto pp2 = FunctionPp("pp2");
  auto node0 = FlowNode("node0", 4, 3)
                   .SetInput(0, data0)
                   .SetInput(1, data1)
                   .SetInput(2, data2)
                   .AddPp(pp1)
                   .MapInput(0, pp1, 2)
                   .MapInput(1, pp1, 1)
                   .MapInput(2, pp1, 0)
                   .MapInput(3, pp2, 0)
                   .MapInput(10, pp2, 0)
                   .MapOutput(0, pp1, 1)
                   .MapOutput(1, pp1, 0)
                   .MapOutput(2, pp2, 0)
                   .MapOutput(10, pp2, 0);

  std::vector<std::string> pp_attrs;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node0);
  ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs);
  auto process_point = dataflow::ProcessPoint();
  auto flag = process_point.ParseFromString(pp_attrs[0]);
  ASSERT_TRUE(flag);
  ASSERT_EQ(process_point.name(), "pp1");
  ASSERT_EQ(process_point.type(), dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  ASSERT_EQ(process_point.compile_cfg_file(), "./pp1.json");
  ASSERT_EQ(process_point.funcs_size(), 0);
  ASSERT_EQ(process_point.graphs_size(), 0);
  ASSERT_EQ(process_point.invoke_pps_size(), 0);

  // in_edges check
  ASSERT_EQ(process_point.in_edges_size(), 3);
  auto in_edges0 = process_point.in_edges(0);
  ASSERT_EQ(in_edges0.node_name(), "node0");
  ASSERT_EQ(in_edges0.index(), 2);
  auto in_edges1 = process_point.in_edges(1);
  ASSERT_EQ(in_edges1.node_name(), "node0");
  ASSERT_EQ(in_edges1.index(), 1);
  auto in_edges2 = process_point.in_edges(2);
  ASSERT_EQ(in_edges2.node_name(), "node0");
  ASSERT_EQ(in_edges2.index(), 0);

  // out_edges check
  ASSERT_EQ(process_point.out_edges_size(), 2);
  auto out_edges0 = process_point.out_edges(0);
  ASSERT_EQ(out_edges0.node_name(), "node0");
  ASSERT_EQ(out_edges0.index(), 1);
  auto out_edges1 = process_point.out_edges(1);
  ASSERT_EQ(out_edges1.node_name(), "node0");
  ASSERT_EQ(out_edges1.index(), 0);

  FlowGraph flow_graph("flow_graph");
  std::vector<FlowOperator> inputs_operator{data0, data1, data2};
  std::vector<FlowOperator> outputs_operator{node0};
  std::vector<FlowOperator> empty_flow_ops;
  flow_graph.SetInputs(inputs_operator).SetOutputs(outputs_operator);
  ASSERT_EQ(strcmp(flow_graph.GetName(), "flow_graph"), 0);
  ASSERT_EQ(flow_graph.ToGeGraph().GetName(), "flow_graph");

  FlowGraph flow_graph_out_not_use("flow_graph_out_not_use");
  std::vector<std::pair<FlowOperator, std::vector<size_t>>> output_indexes;
  std::vector<size_t> part_out{0};
  output_indexes.emplace_back(node0, part_out);
  flow_graph_out_not_use.SetInputs(inputs_operator).SetOutputs(output_indexes);
  ASSERT_EQ(strcmp(flow_graph_out_not_use.GetName(), "flow_graph_out_not_use"), 0);

  FlowGraph flow_graph2(nullptr);
  flow_graph2.SetInputs(empty_flow_ops).SetOutputs(empty_flow_ops);
  ASSERT_EQ(flow_graph2.GetName(), nullptr);

  FlowGraph flow_graph3("flow_graph");
  flow_graph3.SetInputs(empty_flow_ops).SetOutputs(empty_flow_ops);
  ASSERT_EQ(flow_graph3.ToGeGraph().GetName(), "flow_graph");
}

TEST_F(FlowGraphUTest, DflowInvokePp) {
  auto data0 = FlowData("Data0", 0);
  auto data1 = FlowData("Data1", 1);
  auto data2 = FlowData("Data2", 2);

  ge::Graph ge_graph("ge_graph");
  GraphBuilder graph_build = [ge_graph]() { return ge_graph; };
  // GraphBuilder graph_build2 = []() { return ge::Graph("ge_graph2"); };
  FlowGraph flow_graph("flow_graph");
  FlowGraphBuilder flow_graph_build = [flow_graph]() { return flow_graph; };
  auto graphPp1 = GraphPp("graphPp_1", graph_build).SetCompileConfig("./graph.json");
  auto flowGraphPp2 = FlowGraphPp("flowGraphPp_2", flow_graph_build).SetCompileConfig("./flowgraph.json");
  auto pp1 = FunctionPp("pp1")
                 .SetCompileConfig("./pp1.json")
                 .AddInvokedClosure("graph1", graphPp1)
                 .AddInvokedClosure("graph2", flowGraphPp2);
  auto node0 = FlowNode("node0", 3, 2).SetInput(0, data0).SetInput(1, data1).SetInput(2, data2).AddPp(pp1);
  std::vector<std::string> pp_attrs;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node0);
  ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs);
  auto process_point = dataflow::ProcessPoint();
  auto flag = process_point.ParseFromString(pp_attrs[0]);
  ASSERT_TRUE(flag);
  ASSERT_EQ(process_point.name(), "pp1");
  ASSERT_EQ(process_point.type(), dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  ASSERT_EQ(process_point.compile_cfg_file(), "./pp1.json");
  ASSERT_EQ(process_point.invoke_pps_size(), 2);

  auto invoke_pps = process_point.invoke_pps();
  auto invoke_pp0 = invoke_pps["graph1"];
  ASSERT_EQ(invoke_pp0.name(), "graphPp_1");
  ASSERT_EQ(invoke_pp0.type(), dataflow::ProcessPoint_ProcessPointType_GRAPH);
  ASSERT_EQ(invoke_pp0.compile_cfg_file(), "./graph.json");
  ASSERT_EQ(invoke_pp0.graphs(0), "graphPp_1");

  auto invoke_pp1 = invoke_pps["graph2"];
  ASSERT_EQ(invoke_pp1.name(), "flowGraphPp_2");
  ASSERT_EQ(invoke_pp1.type(), dataflow::ProcessPoint_ProcessPointType_FLOW_GRAPH);
  ASSERT_EQ(invoke_pp1.compile_cfg_file(), "./flowgraph.json");
  ASSERT_EQ(invoke_pp1.graphs(0), "flowGraphPp_2");
}

TEST_F(FlowGraphUTest, MapInputAndMapOutputFailed) {
  auto data0 = FlowData("Data0", 0);
  auto data1 = FlowData("Data1", 1);
  auto data2 = FlowData("Data2", 2);
  auto pp1 = FunctionPp("pp1").SetCompileConfig("./pp1.json");
  auto node0 = FlowNode("node0", 3, 2)
                   .SetInput(0, data0)
                   .SetInput(1, data1)
                   .SetInput(2, data2)
                   .AddPp(pp1)
                   .MapInput(0, pp1, 0)
                   .MapInput(1, pp1, 0)
                   .MapInput(2, pp1, 0)
                   .MapOutput(0, pp1, 0)
                   .MapOutput(1, pp1, 0)
                   .MapOutput(2, pp1, 0);
  std::vector<std::string> pp_attrs;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node0);
  ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs);
  dataflow::ProcessPoint process_point;
  auto flag = process_point.ParseFromString(pp_attrs[0]);
  ASSERT_TRUE(flag);
}

TEST_F(FlowGraphUTest, FlowNode_FlowNodeImpl_nullptr) {
  auto pp = FunctionPp("func_pp");
  auto node = FlowNode(nullptr, 0, 0).AddPp(pp).MapInput(0, pp, 0).MapOutput(0, pp, 0);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node);
  std::vector<std::string> pp_attrs;
  ASSERT_FALSE(ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs));
  node.SetBalanceGather();
  bool attr_value = false;
  ASSERT_FALSE(ge::AttrUtils::GetBool(op_desc, ATTR_NAME_BALANCE_GATHER, attr_value));
  node.SetBalanceScatter();
  ASSERT_FALSE(ge::AttrUtils::GetBool(op_desc, ATTR_NAME_BALANCE_SCATTER, attr_value));
}

namespace {
class StubProcessPoint : public ProcessPoint {
 public:
  StubProcessPoint(const char_t *name, ProcessPointType type) : ProcessPoint(name, type) {}
  void Serialize(ge::AscendString &str) const override {
    return;
  }
};
}  // namespace

TEST_F(FlowGraphUTest, FlowNode_Invalid_Pp) {
  auto pp = FunctionPp(nullptr);
  auto node = FlowNode("node", 1, 1).AddPp(pp).MapInput(0, pp, 0).MapOutput(0, pp, 0);
  auto stub_pp = StubProcessPoint("stub_pp", ProcessPointType::FUNCTION);
  node.AddPp(stub_pp);
  stub_pp = StubProcessPoint("stub_pp", ProcessPointType::GRAPH);
  node.AddPp(stub_pp);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node);
  std::vector<std::string> pp_attrs;
  ASSERT_FALSE(ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs));
}

TEST_F(FlowGraphUTest, FlowNode_MapInput_Failed) {
  auto pp = FunctionPp("pp");
  auto node = FlowNode("node", 1, 1).AddPp(pp);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node);
  DataFlowInputAttr attr{DataFlowAttrType::INVALID, nullptr};
  node.MapInput(0, pp, 0, {attr});
  auto input_desc = op_desc->MutableInputDesc(0);
  input_desc->SetDataType(DT_UNDEFINED);
  input_desc->SetFormat(FORMAT_RESERVED);
  node.MapInput(0, pp, 0);
  std::vector<std::string> pp_attrs;
  ASSERT_TRUE(ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs));
  dataflow::ProcessPoint process_point;
  auto flag = process_point.ParseFromString(pp_attrs[0]);
  ASSERT_TRUE(flag);
  ASSERT_EQ(process_point.in_edges_size(), 0);
}

TEST_F(FlowGraphUTest, FlowNode_AddPp_Failed) {
  auto pp = GraphPp("graphpp", nullptr);
  auto node = FlowNode("node", 1, 1).AddPp(pp);
  std::vector<std::string> pp_attrs;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node);
  ASSERT_FALSE(ge::AttrUtils::GetListStr(op_desc, "_dflow_process_points", pp_attrs));
}

TEST_F(FlowGraphUTest, FlowNode_SetBalanceScatter_Success) {
  auto node = FlowNode("node", 1, 1);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node);
  bool attr_value = false;
  ASSERT_FALSE(ge::AttrUtils::GetBool(op_desc, ATTR_NAME_BALANCE_SCATTER, attr_value));
  node.SetBalanceScatter();
  ASSERT_TRUE(ge::AttrUtils::GetBool(op_desc, ATTR_NAME_BALANCE_SCATTER, attr_value));
  EXPECT_TRUE(attr_value);
}

TEST_F(FlowGraphUTest, FlowNode_SetBalanceGather_Success) {
  auto node = FlowNode("node", 1, 1);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node);
  bool attr_value = false;
  ASSERT_FALSE(ge::AttrUtils::GetBool(op_desc, ATTR_NAME_BALANCE_GATHER, attr_value));
  node.SetBalanceGather();
  ASSERT_TRUE(ge::AttrUtils::GetBool(op_desc, ATTR_NAME_BALANCE_GATHER, attr_value));
  EXPECT_TRUE(attr_value);
}

TEST_F(FlowGraphUTest, FlowNode_SetBalanceGather_conflict_with_scatter) {
  auto node = FlowNode("node", 1, 1);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node);
  node.SetBalanceScatter();
  node.SetBalanceGather();
  bool is_scatter = false;
  bool is_gather = false;
  ASSERT_TRUE(ge::AttrUtils::GetBool(op_desc, ATTR_NAME_BALANCE_SCATTER, is_scatter));
  ASSERT_FALSE(ge::AttrUtils::GetBool(op_desc, ATTR_NAME_BALANCE_GATHER, is_gather));
  EXPECT_TRUE(is_scatter);
  EXPECT_FALSE(is_gather);
}

TEST_F(FlowGraphUTest, FlowNode_SetBalanceScatter_conflict_with_gather) {
  auto node = FlowNode("node", 1, 1);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node);
  node.SetBalanceGather();
  node.SetBalanceScatter();
  bool is_scatter = false;
  bool is_gather = false;
  ASSERT_FALSE(ge::AttrUtils::GetBool(op_desc, ATTR_NAME_BALANCE_SCATTER, is_scatter));
  ASSERT_TRUE(ge::AttrUtils::GetBool(op_desc, ATTR_NAME_BALANCE_GATHER, is_gather));
  EXPECT_FALSE(is_scatter);
  EXPECT_TRUE(is_gather);
}

TEST_F(FlowGraphUTest, FlowGraph_FlowGraphImpl_nullptr) {
  auto data0 = FlowData("Data0", 0);
  auto flow_node = FlowNode("FlowNode", 2, 1);
  flow_node.SetInput(2, data0);
  auto flow_graph = FlowGraph(nullptr);
  flow_graph.SetInputs({data0})
      .SetOutputs({flow_node})
      .SetInputsAlignAttrs(128, 10 * 1000, true)
      .SetContainsNMappingNode(true).SetExceptionCatch(true);
  ASSERT_EQ(flow_graph.ToGeGraph().IsValid(), false);

  std::vector<std::pair<FlowOperator, std::vector<size_t>>> output_indexes;
  std::vector<size_t> part_out{0};
  output_indexes.emplace_back(flow_node, part_out);
  flow_graph.SetInputs({data0}).SetOutputs(output_indexes);
  ASSERT_EQ(flow_graph.ToGeGraph().IsValid(), false);
}

TEST_F(FlowGraphUTest, SetContainsNMappingNode_test) {
  auto data0 = FlowData("Data0", 0);
  auto flow_node = FlowNode("FlowNode", 1, 1);
  flow_node.SetInput(0, data0);
  auto flow_graph = FlowGraph("flow_graph");
  flow_graph.SetInputs({data0}).SetOutputs({flow_node});
  const auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());
  bool contains_n_mapping_node = false;
  ASSERT_FALSE(
      ge::AttrUtils::GetBool(compute_graph, ATTR_NAME_DATA_FLOW_CONTAINS_N_MAPPING_NODE, contains_n_mapping_node));
  flow_graph.SetContainsNMappingNode(false);
  ASSERT_TRUE(
      ge::AttrUtils::GetBool(compute_graph, ATTR_NAME_DATA_FLOW_CONTAINS_N_MAPPING_NODE, contains_n_mapping_node));
  ASSERT_FALSE(contains_n_mapping_node);
  flow_graph.SetContainsNMappingNode(true);
  ASSERT_TRUE(
      ge::AttrUtils::GetBool(compute_graph, ATTR_NAME_DATA_FLOW_CONTAINS_N_MAPPING_NODE, contains_n_mapping_node));
  ASSERT_TRUE(contains_n_mapping_node);
}

TEST_F(FlowGraphUTest, SetInputsAlignAttrs_test) {
  auto data0 = FlowData("Data0", 0);
  auto flow_node = FlowNode("FlowNode", 1, 1);
  flow_node.SetInput(0, data0);
  auto flow_graph = FlowGraph("flow_graph");
  flow_graph.SetInputs({data0}).SetOutputs({flow_node});
  const auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());
  int64_t int_value = 0;
  bool bool_value = false;
  ASSERT_FALSE(ge::AttrUtils::GetInt(compute_graph, ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM, int_value));
  ASSERT_FALSE(ge::AttrUtils::GetInt(compute_graph, ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT, int_value));
  ASSERT_FALSE(ge::AttrUtils::GetBool(compute_graph, ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_DROPOUT, bool_value));
  uint32_t align_max_cache_num = 128;
  int32_t align_timeout = 10 * 1000;
  bool dropout_when_not_align = true;
  flow_graph.SetInputsAlignAttrs(align_max_cache_num, align_timeout, dropout_when_not_align);
  ASSERT_TRUE(ge::AttrUtils::GetInt(compute_graph, ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM, int_value));
  EXPECT_EQ(int_value, align_max_cache_num);
  ASSERT_TRUE(ge::AttrUtils::GetInt(compute_graph, ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT, int_value));
  EXPECT_EQ(int_value, align_timeout);
  ASSERT_TRUE(ge::AttrUtils::GetBool(compute_graph, ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_DROPOUT, bool_value));
  ASSERT_EQ(bool_value, dropout_when_not_align);
}

TEST_F(FlowGraphUTest, SetExceptionCatch_test) {
  auto data0 = FlowData("Data0", 0);
  auto flow_node = FlowNode("FlowNode", 1, 1);
  flow_node.SetInput(0, data0);
  auto flow_graph = FlowGraph("flow_graph");
  flow_graph.SetInputs({data0}).SetOutputs({flow_node});
  const auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());
  bool enable_exception_catch = false;
  ASSERT_FALSE(
      ge::AttrUtils::GetBool(compute_graph, ATTR_NAME_DATA_FLOW_ENABLE_EXCEPTION_CATCH, enable_exception_catch));
  flow_graph.SetExceptionCatch(false);
  ASSERT_TRUE(
      ge::AttrUtils::GetBool(compute_graph, ATTR_NAME_DATA_FLOW_ENABLE_EXCEPTION_CATCH, enable_exception_catch));
  ASSERT_FALSE(enable_exception_catch);
  flow_graph.SetExceptionCatch(true);
  ASSERT_TRUE(
      ge::AttrUtils::GetBool(compute_graph, ATTR_NAME_DATA_FLOW_ENABLE_EXCEPTION_CATCH, enable_exception_catch));
  ASSERT_TRUE(enable_exception_catch);
}
}  // namespace ge
