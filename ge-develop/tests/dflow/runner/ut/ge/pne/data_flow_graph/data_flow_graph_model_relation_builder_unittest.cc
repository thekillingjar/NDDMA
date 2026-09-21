/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "dflow/compiler/data_flow_graph/data_flow_graph.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph_model_relation_builder.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "proto/dflow.pb.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "dflow/flow_graph/data_flow_attr_define.h"

using namespace testing;
namespace ge {
class DataFlowGraphModelRelationBuilderTest : public Test {
 protected:
  void SetUp() {
    std::string cmd = "mkdir -p temp; cd temp; touch libtest.so";
    (void) system(cmd.c_str());
    std::ofstream cmakefile("./temp/CMakeLists.txt");
    {
      cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
      // Prevent cmake from testing the toolchain
      cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
      cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
      cmakefile << "project(test)\n";
      cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
      cmakefile << "execute_process(\n";
      cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so ${RELEASE_DIR}\n";
      cmakefile << ")\n";
      cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
      cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";
    }
  }

  void TearDown() {
    std::string cmd = "rm -rf temp";
    (void) system(cmd.c_str());
  }
};

TEST_F(DataFlowGraphModelRelationBuilderTest, BuildFromDataFlowGraph_SUCCESS) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto net_output = OP_CFG("NetOutput").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("net_output", net_output));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 2},
        {"output_num", 2},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  AttrUtils::SetListStr(node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);

  auto pp1 = dataflow::ProcessPoint();
  pp1.set_name("pp1");
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  std::string pp1_config_file = "./pp1_config.json";
  {
    nlohmann::json pp1_cfg_json = {
        {"inputs_tensor_desc",
         {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}, {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}}};
    std::ofstream json_file(pp1_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  pp1.set_compile_cfg_file(pp1_config_file);
  pp1.add_graphs("pp1");
  std::string pp1_str;
  pp1.SerializeToString(&pp1_str);
  std::vector<std::string> pp1_attr{pp1_str};
  auto node1 = root_graph->FindNode("node1");
  AttrUtils::SetListStr(node1->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp1_attr);

  DEF_GRAPH(sub_graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto sub_graph = GraphUtilsEx::GetComputeGraph(ToGeGraph(sub_graph_def));
  (void)AttrUtils::SetStr(sub_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  sub_graph->SetParentNode(node1);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetName("pp1");
  root_graph->AddSubgraph("pp1", sub_graph);

  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);
  DataFlowGraphModelRelationBuilder builder;
  std::unique_ptr<ModelRelation> model_relation;
  EXPECT_EQ(builder.BuildFromDataFlowGraph(data_flow_graph, model_relation), SUCCESS);
  remove(pp0_config_file.c_str());
  remove(pp1_config_file.c_str());
}

TEST_F(DataFlowGraphModelRelationBuilderTest, BuildFromDataFlowGraph_OutputNotUse) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp1 = dataflow::ProcessPoint();
  pp1.set_name("pp1");
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  std::string pp1_config_file = "./pp1_config.json";
  {
    nlohmann::json pp1_cfg_json = {
        {"inputs_tensor_desc",
         {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}, {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}}};
    std::ofstream json_file(pp1_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  pp1.set_compile_cfg_file(pp1_config_file);
  pp1.add_graphs("pp1");
  std::string pp1_str;
  pp1.SerializeToString(&pp1_str);
  std::vector<std::string> pp1_attr{pp1_str};
  auto node1 = root_graph->FindNode("node0");
  AttrUtils::SetListStr(node1->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp1_attr);

  DEF_GRAPH(sub_graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto sub_graph = GraphUtilsEx::GetComputeGraph(ToGeGraph(sub_graph_def));
  (void)AttrUtils::SetStr(sub_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  sub_graph->SetParentNode(node1);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetName("pp1");
  root_graph->AddSubgraph("pp1", sub_graph);

  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);
  DataFlowGraphModelRelationBuilder builder;
  std::unique_ptr<ModelRelation> model_relation = nullptr;
  EXPECT_EQ(builder.BuildFromDataFlowGraph(data_flow_graph, model_relation), SUCCESS);
  EXPECT_EQ(model_relation->endpoints.size(), 3);
  EXPECT_EQ(model_relation->endpoints[2].GetEndpointType(), EndpointType::kDummyQueue);
  remove(pp1_config_file.c_str());
}
}  // namespace ge
