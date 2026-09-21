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
#include <gtest/gtest.h>
#include "nlohmann/json.hpp"
#include "dflow/compiler/data_flow_graph/data_flow_graph_auto_deployer.h"
#include "graph/utils/graph_utils_ex.h"
#include "flow_graph/data_flow.h"
#include "graph/ge_local_context.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph.h"
#include "dflow/flow_graph/data_flow_attr_define.h"

using namespace testing;
namespace ge {
namespace {
static const std::string kDeployInfoFilePrefix = "deploy_info_file;";
Graph BuildSubGraph(const std::string &name) {
  DEF_GRAPH(tmp_graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto compute_graph = ToComputeGraph(tmp_graph_def);
  (void)AttrUtils::SetStr(compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  compute_graph->SetName(name);
  return GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
}
Graph BuildDataFlowGraphWithHostUdfCallNn(const std::string &name, const std::string &udf_pp_name,
                                          const std::string &udf_pp_config, const std::string &invoked_graph_pp_name,
                                          const std::string &invoked_graph_pp_config) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("node0", 2, 2).SetInput(0, data0).SetInput(1, data1);

  auto invoked_graph_pp0 = dflow::GraphPp(invoked_graph_pp_name.c_str(), []() {
                             return BuildSubGraph("invoked_graph_pp0");
                           }).SetCompileConfig(invoked_graph_pp_config.c_str());
  // function pp
  auto host_udf_pp = dflow::FunctionPp(udf_pp_name.c_str())
                         .SetCompileConfig(udf_pp_config.c_str())
                         .AddInvokedClosure("invoke_graph", invoked_graph_pp0);
  node0.AddPp(host_udf_pp);
  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph(name.c_str());
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph.ToGeGraph();
}

void BuilGraphProcessPoint(dataflow::ProcessPoint &pp0, const std::string &name) {
  pp0.set_name(name);
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {
        {"inputs_tensor_desc",
         {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}, {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);
}

ComputeGraphPtr BuildSubGraphForGraphPoint() {
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
  return sub_graph;
}

ComputeGraphPtr BuildFlowGraphWithOneGraphPoint() {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");

  auto pp0 = dataflow::ProcessPoint();
  BuilGraphProcessPoint(pp0, "pp0");
  pp0.add_graphs("pp0");
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  AttrUtils::SetListStr(node0->GetOpDesc(), "_dflow_process_points", pp0_attr);
  auto sub_graph = BuildSubGraphForGraphPoint();
  sub_graph->SetParentNode(node0);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetName("pp0");
  root_graph->AddSubgraph("pp0", sub_graph);
  return root_graph;
}

ComputeGraphPtr BuildFlowGraphWithUdfCallNnAndFlowGraph() {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
  };

  DEF_GRAPH(pp0_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("Add", add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("Add", add));
  };

  DEF_GRAPH(pp1_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node1", node1));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  AttrUtils::SetBool(root_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true);
  auto pp0 = dataflow::ProcessPoint();
  auto pp1 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp1.set_name("pp1");
  pp0.add_graphs("pp0");
  pp1.add_graphs("pp1");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_FLOW_GRAPH);
  pp0.set_compile_cfg_file("");
  pp1.set_compile_cfg_file("");
  auto pp3 = dataflow::ProcessPoint();
  pp3.set_name("pp3");
  pp3.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp3_config_file = "./pp0_config.json";
  {
    nlohmann::json pp3_cfg_json = {{"input_num", 2},
                                   {"output_num", 1},
                                   {"built_in_flow_func", true},
                                   {"func_list", {{{"func_name", "_BuiltIn_func"}}}},
                                   {"running_resources_info", {{{"type", "cpu"}, {"num", 2}}}}};
    std::ofstream json_file(pp3_config_file);
    json_file << pp3_cfg_json << std::endl;
  }
  pp3.set_compile_cfg_file(pp3_config_file);
  auto pp3_invoke_pps = pp3.mutable_invoke_pps();
  (*pp3_invoke_pps)["invoke_graph_key"] = pp0;
  (*pp3_invoke_pps)["invoke_flow_graph_key"] = pp1;
  std::string pp3_str;
  pp3.SerializeToString(&pp3_str);
  std::vector<std::string> pp3_attr{pp3_str};
  auto node0 = root_graph->FindNode("node0");
  EXPECT_TRUE(node0 != nullptr);
  AttrUtils::SetListStr(node0->GetOpDesc(), "_dflow_process_points", pp3_attr);

  auto sub_graph_pp0 = ToComputeGraph(pp0_graph);
  auto sub_graph_pp1 = ToComputeGraph(pp1_graph);
  AttrUtils::SetBool(sub_graph_pp1, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true);
  sub_graph_pp0->SetParentNode(node0);
  sub_graph_pp0->SetParentGraph(root_graph);
  sub_graph_pp0->SetName("pp0");
  root_graph->AddSubgraph("pp0", sub_graph_pp0);
  sub_graph_pp1->SetParentNode(node0);
  sub_graph_pp1->SetParentGraph(root_graph);
  sub_graph_pp1->SetName("pp1");
  root_graph->AddSubgraph("pp1", sub_graph_pp1);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  return root_graph;
}

ComputeGraphPtr BuildFlowGraphWithTwoGraphPoints() {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(0, 1)->NODE("node1", node1));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");

  auto pp0 = dataflow::ProcessPoint();
  BuilGraphProcessPoint(pp0, "pp0");
  pp0.add_graphs("pp0");
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  AttrUtils::SetListStr(node0->GetOpDesc(), "_dflow_process_points", pp0_attr);

  auto pp1 = dataflow::ProcessPoint();
  BuilGraphProcessPoint(pp1, "pp1");
  pp1.add_graphs("pp1");
  std::string pp1_str;
  pp1.SerializeToString(&pp1_str);
  std::vector<std::string> pp1_attr{pp1_str};
  auto node1 = root_graph->FindNode("node1");
  AttrUtils::SetListStr(node1->GetOpDesc(), "_dflow_process_points", pp1_attr);

  auto sub_graph1 = BuildSubGraphForGraphPoint();
  sub_graph1->SetParentNode(node0);
  sub_graph1->SetParentGraph(root_graph);
  sub_graph1->SetName("pp0");
  root_graph->AddSubgraph("pp0", sub_graph1);
  auto sub_graph2 = BuildSubGraphForGraphPoint();
  sub_graph2->SetParentNode(node1);
  sub_graph2->SetParentGraph(root_graph);
  sub_graph2->SetName("pp1");
  root_graph->AddSubgraph("pp1", sub_graph2);
  return root_graph;
}


}  // namespace

class DataFlowGraphAutoDeployerTest : public Test {
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
    cmd = "rm -rf ./pp0_config.json";
    (void) system(cmd.c_str());
  }
};

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployerSuccess) {
  DataFlowGraph data_flow_graph(BuildFlowGraphWithOneGraphPoint());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, ""), SUCCESS);
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, "0:0:0:1;"), SUCCESS);
  EXPECT_EQ(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph), SUCCESS);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_deploy_config) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:1:0,0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithOneGraphPoint());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  EXPECT_EQ(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph), SUCCESS);
  remove(cfg_file_name);
  const auto &subgraph_map = data_flow_graph.GetAllSubgraphs();
  ASSERT_EQ(subgraph_map.size(), 1);
  const auto &subgraph = subgraph_map.begin()->second;

  std::string logic_device_id;
  EXPECT_TRUE(AttrUtils::GetStr(subgraph, ATTR_NAME_LOGIC_DEV_ID, logic_device_id));
  EXPECT_EQ(logic_device_id, "0:0:0:0,0:0:1:0");
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_deploy_config_invoke) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "keep_logic_device_order":true,
        "dynamic_schedule_enable": true,
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0",
            "invoke_list":[
              {
                "invoke_name":"invoke_graph_key",
                "logic_device_list":"0:0:1:0",
                "redundant_logic_device_list":"0:0:2:0"
              },
              {
                "invoke_name":"invoke_flow_graph_key",
                "deploy_info_file":"./deploy_info_invoke.json"
              }
            ]
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithUdfCallNnAndFlowGraph());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  EXPECT_EQ(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph), SUCCESS);
  remove(cfg_file_name);
  const auto &subgraph_map = data_flow_graph.GetAllSubgraphs();
  ASSERT_EQ(subgraph_map.size(), 3);
  for (const auto &subgraph_iter : subgraph_map) {
    const auto &subgraph = subgraph_iter.second;
    std::string logic_device_id;
    EXPECT_TRUE(AttrUtils::GetStr(subgraph, ATTR_NAME_LOGIC_DEV_ID, logic_device_id));
    if (subgraph_iter.first == "pp0") {
      EXPECT_EQ(logic_device_id, "0:0:1:0,0:0:2:0");
    } else {
      EXPECT_EQ(logic_device_id, "0:0:0:0");
    }
  }
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_deploy_config_invoke_test_redundant) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "keep_logic_device_order":true,
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0",
            "redundant_logic_device_list":"0:0:1~2:0",
            "invoke_list":[
              {
                "invoke_name":"invoke_graph_key",
                "logic_device_list":"0:0:1:0",
                "redundant_logic_device_list":"0:0:2:0"
              },
              {
                "invoke_name":"invoke_flow_graph_key",
                "deploy_info_file":"./deploy_info_invoke.json"
              }
            ]
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithUdfCallNnAndFlowGraph());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  EXPECT_EQ(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph), SUCCESS);
  remove(cfg_file_name);
  const auto &subgraph_map = data_flow_graph.GetAllSubgraphs();
  ASSERT_EQ(subgraph_map.size(), 3);
  for (const auto &subgraph_iter : subgraph_map) {
    const auto &subgraph = subgraph_iter.second;
    std::string logic_device_id;
    EXPECT_TRUE(AttrUtils::GetStr(subgraph, ATTR_NAME_LOGIC_DEV_ID, logic_device_id));
    if (subgraph_iter.first == "pp0") {
      EXPECT_EQ(logic_device_id, "0:0:1:0");
    } else {
      EXPECT_EQ(logic_device_id, "0:0:0:0");
    }
  }
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_deploy_config_invoke_only_redundant) {
  constexpr const char *cfg_file_name = "./deploy_info;test.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0",
            "invoke_list":[
              {
                "invoke_name":"invoke_flow_graph_key",
                "redundant_logic_device_list":"0:0:1:0"
              }
            ]
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithUdfCallNnAndFlowGraph());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_deploy_config_invoke_deploy_and_logic_device) {
  constexpr const char *cfg_file_name = "./deploy_info;test.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0",
            "invoke_list":[
              {
                "invoke_name":"invoke_flow_graph_key",
                "deploy_info_file":"./deploy_info_invoke.json",
                "logic_device_list":"0:0:1:0"
              }
            ]
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithUdfCallNnAndFlowGraph());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_deploy_config_invoke_repeat_invoke) {
  constexpr const char *cfg_file_name = "./deploy_info;test.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0","node1"],
            "logic_device_list":"0:0:0:0",
            "invoke_list":[
              {
                "invoke_name":"invoke_flow_graph_key",
                "deploy_info_file":"./deploy_info_invoke.json"
              },
              {
                "invoke_name":"invoke_flow_graph_key",
                "logic_device_list":"0:0:1:0"
              }
            ]
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithUdfCallNnAndFlowGraph());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_deploy_config_invoke_multi_instance) {
  constexpr const char *cfg_file_name = "./deploy_info;test.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0~1:0",
            "invoke_list":[
              {
                "invoke_name":"invoke_flow_graph_key",
                "deploy_info_file":"./deploy_info_invoke.json"
              }
            ]
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithUdfCallNnAndFlowGraph());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  EXPECT_NE(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_deploy_config_invoke_diff_num) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0",
            "invoke_list":[
              {
                "invoke_name":"invoke_graph_key",
                "logic_device_list":"0:0:1~2:0"
              },
              {
                "invoke_name":"invoke_flow_graph_key",
                "deploy_info_file":"./deploy_info_invoke.json"
              }
            ]
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithUdfCallNnAndFlowGraph());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  EXPECT_NE(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_deploy_config_invoke_graph_error) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0",
            "invoke_list":[
              {
                "invoke_name":"invoke_graph_key",
                "deploy_info_file":"./deploy_info_invoke.json"
              }
            ]
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithUdfCallNnAndFlowGraph());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  EXPECT_NE(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_invoke_graph_node_name_error) {
  constexpr const char *cfg_file_name = "./deploy_info_test.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:0~1:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithUdfCallNnAndFlowGraph(), "node3/");
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_redundant_deploy_config) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:1:0,0:0:0:0",
            "redundant_logic_device_list":"0:0:2:0"
          }
        ],
        "dynamic_schedule_enable": true
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithOneGraphPoint());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  EXPECT_EQ(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph), SUCCESS);
  remove(cfg_file_name);
  const auto &subgraph_map = data_flow_graph.GetAllSubgraphs();
  ASSERT_EQ(subgraph_map.size(), 1);
  const auto &subgraph = subgraph_map.begin()->second;

  std::string logic_device_id;
  EXPECT_TRUE(AttrUtils::GetStr(subgraph, ATTR_NAME_LOGIC_DEV_ID, logic_device_id));
  EXPECT_EQ(logic_device_id, "0:0:0:0,0:0:1:0,0:0:2:0");
  std::string redundant_logic_device_id;
  EXPECT_TRUE(AttrUtils::GetStr(subgraph, ATTR_NAME_REDUNDANT_LOGIC_DEV_ID, redundant_logic_device_id));
  EXPECT_EQ(redundant_logic_device_id, "0:0:2:0");
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_server_udf_call_nn_heavy_load) {
  setenv("RESOURCE_CONFIG_PATH", "./stub_resource_config_path.json", 0);
  constexpr const char *cfg_file_name = "./deploy_info.json";
  {
    std::ofstream json_file(cfg_file_name);
    std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:1:0"
          }
        ]
      })";
    json_file << content << std::endl;
  }
  constexpr const char *udf_pp_name = "udf_pp";
  constexpr const char *udf_pp_config = "./host_udf_config.json";
  {
    nlohmann::json host_udf_cfg_json = {{"workspace", "./temp"},
                                        {"target_bin", "libtest.so"},
                                        {"input_num", 2},
                                        {"output_num", 2},
                                        {"heavy_load", true},
                                        {"cmakelist_path", "CMakeLists.txt"},
                                        {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(udf_pp_config);
    json_file << host_udf_cfg_json << std::endl;
  }
  constexpr const char *invoked_graph_pp_name = "invoked_graph_pp";
  constexpr const char *invoked_graph_pp_config = "./graph_pp_config.json";
  {
    nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                    {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                     {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}}};
    std::ofstream json_file(invoked_graph_pp_config);
    json_file << pp1_cfg_json << std::endl;
  }

  auto root_graph = BuildDataFlowGraphWithHostUdfCallNn("host_call_nn", udf_pp_name, udf_pp_config,
                                                        invoked_graph_pp_name, invoked_graph_pp_config);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(root_graph);
  (void)AttrUtils::SetStr(compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  DataFlowGraph data_flow_graph(compute_graph);

  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  EXPECT_EQ(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph), SUCCESS);
  const auto &subgraph_map = data_flow_graph.GetAllSubgraphs();
  ASSERT_EQ(subgraph_map.size(), 2);
  for (const auto &subgraph_iter : subgraph_map) {
    const auto &subgraph = subgraph_iter.second;
    std::string logic_device_id;
    EXPECT_TRUE(AttrUtils::GetStr(subgraph, ATTR_NAME_LOGIC_DEV_ID, logic_device_id));
    // heavy load compile logic device is npu, will be change when deploy
    EXPECT_EQ(logic_device_id, "0:0:1:0");
  }

  remove(cfg_file_name);
  remove(udf_pp_config);
  remove(invoked_graph_pp_config);
  unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_server_heavy_load_failed_no_deploy_info) {
  setenv("RESOURCE_CONFIG_PATH", "./stub_resource_config_path.json", 0);
  constexpr const char *udf_pp_name = "udf_pp";
  constexpr const char *udf_pp_config = "./host_udf_config.json";
  {
    nlohmann::json host_udf_cfg_json = {{"workspace", "./temp"},
                                        {"target_bin", "libtest.so"},
                                        {"input_num", 2},
                                        {"output_num", 2},
                                        {"heavy_load", true},
                                        {"cmakelist_path", "CMakeLists.txt"},
                                        {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(udf_pp_config);
    json_file << host_udf_cfg_json << std::endl;
  }
  constexpr const char *invoked_graph_pp_name = "invoked_graph_pp";
  constexpr const char *invoked_graph_pp_config = "./graph_pp_config.json";
  {
    nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                    {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                     {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}}};
    std::ofstream json_file(invoked_graph_pp_config);
    json_file << pp1_cfg_json << std::endl;
  }

  auto root_graph = BuildDataFlowGraphWithHostUdfCallNn("host_call_nn", udf_pp_name, udf_pp_config,
                                                        invoked_graph_pp_name, invoked_graph_pp_config);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(root_graph);
  (void)AttrUtils::SetStr(compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  DataFlowGraph data_flow_graph(compute_graph);

  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);

  std::map<std::string, std::string> graph_options = {};
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, kDeployInfoFilePrefix), SUCCESS);
  remove(udf_pp_config);
  remove(invoked_graph_pp_config);
  unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_batch_deploy_config) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0~1:0~1:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  DataFlowGraph data_flow_graph(BuildFlowGraphWithOneGraphPoint());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);
  
  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  EXPECT_EQ(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph), SUCCESS);
  remove(cfg_file_name);
  const auto &subgraph_map = data_flow_graph.GetAllSubgraphs();
  ASSERT_EQ(subgraph_map.size(), 1);
  const auto &subgraph = subgraph_map.begin()->second;

  std::string logic_device_id;
  EXPECT_TRUE(AttrUtils::GetStr(subgraph, ATTR_NAME_LOGIC_DEV_ID, logic_device_id));
  auto split_logic_device_list = StringUtils::Split(logic_device_id, ',');
  std::set<std::string> logic_device_set(split_logic_device_list.begin(), split_logic_device_list.end());
  std::set<std::string> expect_devce_list = {"0:0:0:0", "0:0:1:0", "0:1:0:0", "0:1:1:0"};
  EXPECT_EQ(logic_device_set, expect_devce_list);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_both_deploy_config) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:0:0,0:0:1:0"
          }
        ],
        "batch_deploy_info": [
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:2:0:0,0:0~1:0~1:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithTwoGraphPoints());
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);
  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  EXPECT_EQ(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph), SUCCESS);
  remove(cfg_file_name);
  const auto &subgraph_map = data_flow_graph.GetAllSubgraphs();
  ASSERT_EQ(subgraph_map.size(), 2);
  {
    auto find_ret = subgraph_map.find("pp0");
    ASSERT_TRUE(find_ret != subgraph_map.cend());
    const auto &subgraph_pp0 = find_ret->second;
    std::string logic_device_id;
    EXPECT_TRUE(AttrUtils::GetStr(subgraph_pp0, ATTR_NAME_LOGIC_DEV_ID, logic_device_id));
    auto split_logic_device_list = StringUtils::Split(logic_device_id, ',');
    std::set<std::string> logic_device_set(split_logic_device_list.begin(), split_logic_device_list.end());
    std::set<std::string> expect_devce_list = {"0:0:0:0", "0:0:1:0"};
    EXPECT_EQ(logic_device_set, expect_devce_list);
  }

  {
    auto find_ret = subgraph_map.find("pp1");
    ASSERT_TRUE(find_ret != subgraph_map.cend());
    const auto &subgraph_pp1 = find_ret->second;
    std::string logic_device_id;
    EXPECT_TRUE(AttrUtils::GetStr(subgraph_pp1, ATTR_NAME_LOGIC_DEV_ID, logic_device_id));
    auto split_logic_device_list = StringUtils::Split(logic_device_id, ',');
    std::set<std::string> logic_device_set(split_logic_device_list.begin(), split_logic_device_list.end());
    std::set<std::string> expect_devce_list = {"0:0:0:0", "0:0:1:0", "0:1:0:0", "0:1:1:0", "0:2:0:0"};
    EXPECT_EQ(logic_device_set, expect_devce_list);
  }
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_repeat_deploy_config) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:0:0,0:0:1:0"
          },
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:0:0,0:0:1:0"
          },
          {
            "flow_node_name":"node1",
            "logic_device_id":"0:0:0:0,0:0:1:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithTwoGraphPoints());
  data_flow_graph.Initialize();

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_mis_part_flow_node_deploy_info) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0~1:0~1:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithTwoGraphPoints());
  data_flow_graph.Initialize();

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_mis_node_deploy_info_with_affinity) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(0, 1)->NODE("node1", node1));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  auto pp1 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp1.set_name("pp1");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 2},
        {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);
  pp1.set_compile_cfg_file(pp0_config_file);
  std::string pp0_str;
  std::string pp1_str;
  pp0.SerializeToString(&pp0_str);
  pp1.SerializeToString(&pp1_str);
  std::vector<std::string> pp0_attr{pp0_str};
  std::vector<std::string> pp1_attr{pp1_str};
  auto node0 = root_graph->FindNode("node0");
  auto node1 = root_graph->FindNode("node1");
  AttrUtils::SetListStr(node0->GetOpDesc(), "_dflow_process_points", pp0_attr);
  AttrUtils::SetListStr(node1->GetOpDesc(), "_dflow_process_points", pp1_attr);
  node1->GetOpDesc()->SetExtAttr("_data_flow_deploy_affinity_node", std::string("node0"));
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0~1:0~1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(root_graph);
  data_flow_graph.Initialize();

  {
    // reset compiler_result for test.
    auto op_desc0 = node0->GetOpDesc();
    auto op_desc1 = node1->GetOpDesc();
    ge::NamedAttrs compile_results_pp0;
    ge::NamedAttrs compile_results_pp1;
    ge::NamedAttrs current_compile_result;
    ge::NamedAttrs runnable_resources_info;
    runnable_resources_info.SetAttr("X86", GeAttrValue::CreateFrom<std::string>("./release.tar.gz"));
    runnable_resources_info.SetAttr("Ascend", GeAttrValue::CreateFrom<std::string>("./release.tar.gz"));
    ge::AttrUtils::SetNamedAttrs(current_compile_result, "_dflow_runnable_resource", runnable_resources_info);
    compile_results_pp0.SetAttr("pp0", GeAttrValue::CreateFrom<ge::NamedAttrs>(current_compile_result));
    compile_results_pp1.SetAttr("pp1", GeAttrValue::CreateFrom<ge::NamedAttrs>(current_compile_result));
    ge::AttrUtils::SetNamedAttrs(op_desc0, "_dflow_compiler_result", compile_results_pp0);
    ge::AttrUtils::SetNamedAttrs(op_desc1, "_dflow_compiler_result", compile_results_pp1);
  }

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph,cfg_path), SUCCESS);
  remove(cfg_file_name);
}


TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_affinity_node_mis_node_deploy_info) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(0, 1)->NODE("node1", node1));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  auto pp1 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp1.set_name("pp1");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 2},
        {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);
  pp1.set_compile_cfg_file(pp0_config_file);
  std::string pp0_str;
  std::string pp1_str;
  pp0.SerializeToString(&pp0_str);
  pp1.SerializeToString(&pp1_str);
  std::vector<std::string> pp0_attr{pp0_str};
  std::vector<std::string> pp1_attr{pp1_str};
  auto node0 = root_graph->FindNode("node0");
  auto node1 = root_graph->FindNode("node1");
  AttrUtils::SetListStr(node0->GetOpDesc(), "_dflow_process_points", pp0_attr);
  AttrUtils::SetListStr(node1->GetOpDesc(), "_dflow_process_points", pp1_attr);
  node1->GetOpDesc()->SetExtAttr("_data_flow_deploy_affinity_node", std::string("node2"));
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0~1:0~1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(root_graph);
  data_flow_graph.Initialize();

  {
    // reset compiler_result for test.
    auto op_desc0 = node0->GetOpDesc();
    auto op_desc1 = node1->GetOpDesc();
    ge::NamedAttrs compile_results_pp0;
    ge::NamedAttrs compile_results_pp1;
    ge::NamedAttrs current_compile_result;
    ge::NamedAttrs runnable_resources_info;
    runnable_resources_info.SetAttr("X86", GeAttrValue::CreateFrom<std::string>("./release.tar.gz"));
    runnable_resources_info.SetAttr("Ascend", GeAttrValue::CreateFrom<std::string>("./release.tar.gz"));
    ge::AttrUtils::SetNamedAttrs(current_compile_result, "_dflow_runnable_resource", runnable_resources_info);
    compile_results_pp0.SetAttr("pp0", GeAttrValue::CreateFrom<ge::NamedAttrs>(current_compile_result));
    compile_results_pp1.SetAttr("pp1", GeAttrValue::CreateFrom<ge::NamedAttrs>(current_compile_result));
    ge::AttrUtils::SetNamedAttrs(op_desc0, "_dflow_compiler_result", compile_results_pp0);
    ge::AttrUtils::SetNamedAttrs(op_desc1, "_dflow_compiler_result", compile_results_pp1);
  }

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_repeat_deploy_config_batch) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:0:0,0:0:1:0"
          }
        ],
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node1"],
            "logic_device_list":"0:0~1:0~1:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithTwoGraphPoints());
  data_flow_graph.Initialize();

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_deploy_config_not_suitable) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
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
        {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  AttrUtils::SetListStr(node0->GetOpDesc(), "_dflow_process_points", pp0_attr);

  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:0,0:0:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(root_graph);
  data_flow_graph.Initialize();
  {
    // reset compiler_result for test.
    auto op_desc = node0->GetOpDesc();
    ge::NamedAttrs compile_results;
    ge::NamedAttrs current_compile_result;
    ge::NamedAttrs runnable_resources_info;
    runnable_resources_info.SetAttr("X86", GeAttrValue::CreateFrom<std::string>("./release.tar.gz"));
    runnable_resources_info.SetAttr("Aarch64", GeAttrValue::CreateFrom<std::string>("./release.tar.gz"));
    ge::AttrUtils::SetNamedAttrs(current_compile_result, "_dflow_runnable_resource", runnable_resources_info);
    compile_results.SetAttr("pp0", GeAttrValue::CreateFrom<ge::NamedAttrs>(current_compile_result));
    ge::AttrUtils::SetNamedAttrs(op_desc, "_dflow_compiler_result", compile_results);
  }

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), FAILED);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_batch_deploy_config_range_limit) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node1"],
            "logic_device_list":"0:1~32767:0~1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  auto root_graph = std::make_shared<ComputeGraph>("test-graph");
  DataFlowGraph data_flow_graph(root_graph);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_num_invalid) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node1"],
            "logic_device_list":"0:-1:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  auto root_graph = std::make_shared<ComputeGraph>("test-graph");
  DataFlowGraph data_flow_graph(root_graph);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  // not support -1
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_batch_deploy_config_range_num_invalid) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node1"],
            "logic_device_list":"0:1~99999999999999999999999999999:0~1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  auto root_graph = std::make_shared<ComputeGraph>("test-graph");
  DataFlowGraph data_flow_graph(root_graph);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_batch_deploy_config_host_format_error) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node1"],
            "logic_device_list":"1:0:-1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  auto root_graph = std::make_shared<ComputeGraph>("test-graph");
  DataFlowGraph data_flow_graph(root_graph);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_batch_deploy_config_ascend_format_error) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node1"],
            "logic_device_list":"aa:bb:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  auto root_graph = std::make_shared<ComputeGraph>("test-graph");
  DataFlowGraph data_flow_graph(root_graph);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_batch_deploy_config_range_over) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node1"],
            "logic_device_list":"0:0~32768:0~1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  auto root_graph = std::make_shared<ComputeGraph>("test-graph");
  DataFlowGraph data_flow_graph(root_graph);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_batch_deploy_config_no_range) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node1"],
            "logic_device_list":"0:1:0,0:0:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  auto root_graph = std::make_shared<ComputeGraph>("test-graph");
  DataFlowGraph data_flow_graph(root_graph);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_batch_deploy_config_range_invalid) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node1"],
            "logic_device_list":"0:0~1:2~1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  auto root_graph = std::make_shared<ComputeGraph>("test-graph");
  DataFlowGraph data_flow_graph(root_graph);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_batch_deploy_config_too_much) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node1"],
            "logic_device_list":"0:0~2048:0~2048"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  auto root_graph = std::make_shared<ComputeGraph>("test-graph");
  DataFlowGraph data_flow_graph(root_graph);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_batch_deploy_config_format_error) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node1"],
            "logic_device_list":"0:0~1:0~"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  auto root_graph = std::make_shared<ComputeGraph>("test-graph");
  DataFlowGraph data_flow_graph(root_graph);

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_NE(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployerFailed) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
  };

  auto root_graph = ToComputeGraph(flow_graph);
  auto node0 = root_graph->FindNode("node0");
  auto op_desc = node0->GetOpDesc();
  DataFlowGraph data_flow_graph(root_graph);
  ge::NamedAttrs compile_results;
  ge::NamedAttrs current_compile_result;
  ge::NamedAttrs runnable_resources_info;
  ge::AttrUtils::SetNamedAttrs(current_compile_result, "_dflow_runnable_resource", runnable_resources_info);
  compile_results.SetAttr("pp0", GeAttrValue::CreateFrom<ge::NamedAttrs>(current_compile_result));
  ge::AttrUtils::SetNamedAttrs(op_desc, "_dflow_compiler_result", compile_results);
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, kDeployInfoFilePrefix), FAILED);
  EXPECT_EQ(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph), FAILED);
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_mem_limit) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:0:0,0:0:0:1"
          }
        ],
        "batch_deploy_info": [
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:2:0,0:0:0~1:0~1"
          }
        ],
        "deploy_mem_info":[
          {
            "std_mem_size":"1024",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:0:0,0:0:1:0"
          },
          {
            "std_mem_size":"2048",
            "shared_mem_size":"2048",
            "logic_device_id":"0:0:2:0,0:0:0~1:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithTwoGraphPoints());
  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), SUCCESS);
  remove(cfg_file_name);
  {
    const auto root_graph = data_flow_graph.GetRootGraph();
    ASSERT_TRUE(root_graph != nullptr);
    const auto &logic_dev_id_to_mem_cfg = root_graph->TryGetExtAttr("_dflow_logic_device_memory_config",
      std::map<std::string, std::pair<uint32_t, uint32_t>>());
    EXPECT_EQ(logic_dev_id_to_mem_cfg.size(), 5);
    const std::pair<uint32_t, uint32_t> pair1 = std::make_pair(1024, 1024);
    const std::pair<uint32_t, uint32_t> pair2 = std::make_pair(2048, 2048);
    auto iter = logic_dev_id_to_mem_cfg.find("0:0:0:0");
    EXPECT_TRUE(iter != logic_dev_id_to_mem_cfg.end());
    EXPECT_EQ(iter->second, pair1);
    iter = logic_dev_id_to_mem_cfg.find("0:0:2:0");
    EXPECT_TRUE(iter != logic_dev_id_to_mem_cfg.end());
    EXPECT_EQ(iter->second, pair2);
    iter = logic_dev_id_to_mem_cfg.find("0:0:0:1");
    EXPECT_TRUE(iter != logic_dev_id_to_mem_cfg.end());
    EXPECT_EQ(iter->second, pair2);
    iter = logic_dev_id_to_mem_cfg.find("0:0:1:1");
    EXPECT_TRUE(iter != logic_dev_id_to_mem_cfg.end());
    EXPECT_EQ(iter->second, pair2);
  }
}

TEST_F(DataFlowGraphAutoDeployerTest, DataFlowGraphAutoDeployer_with_mem_limit_err_config) {
  constexpr const char *cfg_file_name = "./deploy_info.json";
  std::ofstream json_file(cfg_file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:0:0,0:0:0:1"
          }
        ],
        "batch_deploy_info": [
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:2:0,0:0:0~1:0~1"
          }
        ],
        "deploy_mem_info":[
          {
            "std_mem_size":"aa",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:0:0,0:0:1:0"
          },
          {
            "std_mem_size":"2048",
            "shared_mem_size":"2048",
            "logic_device_id":"0:0:2:0,0:0:0~1:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  DataFlowGraph data_flow_graph(BuildFlowGraphWithTwoGraphPoints());

  std::map<std::string, std::string> graph_options = {{"ge.experiment.data_flow_deploy_info_path", cfg_file_name}};
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::string cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), FAILED);
  remove(cfg_file_name);

  std::ofstream json_file1(cfg_file_name);
  content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:0:0"
          }
        ],
        "batch_deploy_info": [
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:2:0,0:0:0~1:0~1"
          }
        ],
        "deploy_mem_info":[
          {
            "std_mem_size":"99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:0"
          },
          {
            "std_mem_size":"2048",
            "shared_mem_size":"2048",
            "logic_device_id":"0:0:2:0,0:0:0~1:1"
          }
        ]
      })";
  json_file1 << content << std::endl;
  json_file1.close();
  cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), FAILED);
  remove(cfg_file_name);

  std::ofstream json_file2(cfg_file_name);
  content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:0:0,0:0:0:1"
          }
        ],
        "batch_deploy_info": [
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:2:0,0:0:0~1:0~1"
          }
        ],
        "deploy_mem_info":[
          {
            "std_mem_size":"1024",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:0!:0"
          },
          {
            "std_mem_size":"2048",
            "shared_mem_size":"2048",
            "logic_device_id":"0:0:2:0,0:0:0!~1:1"
          }
        ]
      })";
  json_file2 << content << std::endl;
  json_file2.close();
  cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), FAILED);
  remove(cfg_file_name);

  std::ofstream json_file3(cfg_file_name);
  content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:0:0,0:0:0:1"
          }
        ],
        "batch_deploy_info": [
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:2:0,0:0:0~1:0~1"
          }
        ],
        "deploy_mem_info":[
          {
            "std_mem_size":"1024",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:0:0"
          }
        ]
      })";
  json_file3 << content << std::endl;
  json_file3.close();
  cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), FAILED);
  remove(cfg_file_name);

  std::ofstream json_file4(cfg_file_name);
  content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:0:0,0:0:0:1"
          }
        ],
        "batch_deploy_info": [
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:2:0,0:0:0~1:0~1"
          }
        ],
        "deploy_mem_info":[
          {
            "std_mem_size":"1024",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:0:0"
          },
          {
            "std_mem_size":"2048",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:0:0"
          }
        ]
      })";
  json_file4 << content << std::endl;
  json_file4.close();
  cfg_path = kDeployInfoFilePrefix + cfg_file_name;
  EXPECT_EQ(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, cfg_path), FAILED);
  remove(cfg_file_name);
}
}  // namespace ge
