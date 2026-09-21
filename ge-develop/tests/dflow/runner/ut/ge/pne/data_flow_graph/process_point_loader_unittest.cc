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
#include "dflow/compiler/data_flow_graph/process_point_loader.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "proto/dflow.pb.h"
#include "graph/utils/graph_utils.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "flow_graph/data_flow.h"
#include "graph/utils/graph_utils_ex.h"
#include "common/ge_common/scope_guard.h"

using namespace testing;
namespace ge {
class ProcessPointLoaderTest : public Test {
 public:
  static void PrepareForCacheConfig(bool cache_manual_check, bool cache_debug_mode) {
    (void)system("mkdir -p ./compile_cache_dir");
    std::string cache_config_file = "./compile_cache_dir/cache.conf";
    {
      nlohmann::json cfg_json = {
                                  {"cache_manual_check", cache_manual_check},
                                  {"cache_debug_mode", cache_debug_mode}};
      std::ofstream json_file(cache_config_file);
      json_file << cfg_json << std::endl;
    }
    (void)system("mkdir -p ./compile_cache_dir/graph1/pp0");
    std::string buildinfo_file = "./compile_cache_dir/graph1/pp0/buildinfo";
    {
      std::map<string, string> bin_info;
      bin_info["Ascend"] = "stub_path";
      std::map<string, int64_t> res_info;
      res_info["Ascend"] = 100;
      nlohmann::json buildinfo_json = {
                                        {"bin_info", bin_info},
                                        {"resource_info", res_info}};
      std::ofstream json_file(buildinfo_file);
      json_file << buildinfo_json << std::endl;
    }
  }

  static void RemoveCacheConfig() {
    (void)system("rm -fr ./compile_cache_dir");
  }
 protected:
  void SetUp() {
    std::string cmd = "mkdir -p temp; cd temp; touch libtest.so";
    (void)system(cmd.c_str());
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
    (void)system(cmd.c_str());
  }
};

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_SUCCESS) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 2},
                                   {"output_num", 2},
                                   {"func_list", {{{"func_name", "func1"}, {"stream_input", true}}}},
                                   {"running_resources_info", {{{"type", "cpu"}, {"num", 2}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);

  DataFlowGraph data_flow_graph(root_graph, "", true);
  auto node0_ptr = root_graph->FindNode("node0");
  ASSERT_TRUE(node0_ptr != nullptr);
  auto op_desc = node0_ptr->GetOpDesc();
  ASSERT_TRUE(op_desc != nullptr);
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_ESCHED_EVENT_PRIORITY, 3));
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_ESCHED_EVENT_PRIORITY, 3));
  EXPECT_NE(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node0_ptr), SUCCESS);
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0));
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0));
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node0_ptr), SUCCESS);
  auto pp1 = dataflow::ProcessPoint();
  pp1.set_is_built_in(true);
  pp1.set_name("Node1__BuiltIn_TimeBatch_0");
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  dataflow::ProcessFunc process_func;
  process_func.set_name("_BuiltIn_TimeBatch");
  process_func.add_in_index(0);
  process_func.add_out_index(0);
  auto process_point_attrs = pp1.mutable_attrs();
  proto::AttrDef window;
  window.set_i(1);
  (*process_point_attrs)["window"] = window;
  auto node1_ptr = root_graph->FindNode("node1");
  EXPECT_TRUE(node1_ptr != nullptr);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(pp1, data_flow_graph, node1_ptr), SUCCESS);
  remove(pp0_config_file.c_str());
  DEF_GRAPH(then_branch) {
    auto data = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto net_output = OP_CFG("NetOutput").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("then_arg_0", data)->NODE("then_Node_Output", net_output));
  };
  DEF_GRAPH(else_branch) {
    auto data = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto neg_op = OP_CFG("Neg").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto net_output = OP_CFG("NetOutput").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("else_arg_0", data)->NODE("neg", neg_op)->NODE("else_Node_Output", net_output));
  };
  auto then_graph = ToComputeGraph(then_branch);
  auto else_graph = ToComputeGraph(else_branch);
  DEF_GRAPH(graph_pp_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto if_op = OP_CFG("If").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("If");
    if_op->MutableOutputDesc(0)->SetShape(GeShape({1, 2, 3}));
    if_op->RegisterSubgraphIrName("then_branch", SubgraphType::kStatic);
    if_op->RegisterSubgraphIrName("else_branch", SubgraphType::kStatic);
    if_op->AddSubgraphName(then_graph->GetName());
    if_op->SetSubgraphInstanceName(0, then_graph->GetName());
    if_op->AddSubgraphName(else_graph->GetName());
    if_op->SetSubgraphInstanceName(1, else_graph->GetName());
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(if_op));
  };
  auto node2_ptr = root_graph->FindNode("node2");
  EXPECT_TRUE(node2_ptr != nullptr);

  auto sub_graph_pp0 = ToComputeGraph(graph_pp_def);
  sub_graph_pp0->SetParentNode(node2_ptr);
  sub_graph_pp0->SetParentGraph(root_graph);
  sub_graph_pp0->SetName("graph_pp0");
  root_graph->AddSubgraph("graph_pp0", sub_graph_pp0);

  auto if_node = sub_graph_pp0->FindFirstNodeMatchType("If");
  EXPECT_TRUE(if_node != nullptr);
  then_graph->SetParentNode(if_node);
  then_graph->SetParentGraph(sub_graph_pp0);
  else_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(sub_graph_pp0);
  sub_graph_pp0->TopologicalSorting();
  root_graph->AddSubgraph("then_branch", then_graph);
  root_graph->AddSubgraph("else_branch", else_graph);

  auto graph_pp0 = dataflow::ProcessPoint();
  graph_pp0.set_name("graph_pp0");
  graph_pp0.add_graphs("graph_pp0");
  graph_pp0.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  EXPECT_EQ(AttrUtils::SetBool(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR, true), true);
  EXPECT_EQ(AttrUtils::SetInt(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR_DEPTH, 128), true);
  EXPECT_EQ(AttrUtils::SetStr(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, "FIFO"), true);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(graph_pp0, data_flow_graph, node2_ptr), SUCCESS);
}

TEST_F(ProcessPointLoaderTest, LoadFuncProcessPointFromCache_SUCCESS) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 2},
                                   {"output_num", 2},
                                   {"func_list", {{{"func_name", "func1"}}}},
                                   {"running_resources_info", {{{"type", "cpu"}, {"num", 2}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);

  auto options_back = GetThreadLocalContext().GetAllGraphOptions();
  auto options = options_back;
  options[OPTION_GRAPH_COMPILER_CACHE_DIR] = "./compile_cache_dir";
  options[OPTION_GRAPH_KEY] = "graph1";
  GetThreadLocalContext().SetGraphOption(options);
  PrepareForCacheConfig(true, true);
  GE_MAKE_GUARD(recover, ([this, options_back](){
    RemoveCacheConfig();
    GetThreadLocalContext().SetGraphOption(options_back);
  }));

  DataFlowGraph data_flow_graph(root_graph, "", true);
  auto node0_ptr = root_graph->FindNode("node0");
  ASSERT_TRUE(node0_ptr != nullptr);
  auto op_desc = node0_ptr->GetOpDesc();
  ASSERT_TRUE(op_desc != nullptr);
  ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0));
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node0_ptr), SUCCESS);

  remove(pp0_config_file.c_str());
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_builtin_func) {
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
    nlohmann::json pp0_cfg_json = {{"input_num", 2},
                                   {"output_num", 1},
                                   {"built_in_flow_func", true},
                                   {"func_list", {{{"func_name", "_BuiltIn_func"}}}},
                                   {"running_resources_info", {{{"type", "cpu"}, {"num", 2}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);

  DataFlowGraph data_flow_graph(root_graph);
  auto node0_ptr = root_graph->FindNode("node0");
  ASSERT_TRUE(node0_ptr != nullptr);
  auto op_desc = node0_ptr->GetOpDesc();
  ASSERT_TRUE(op_desc != nullptr);
  (void)AttrUtils::SetInt(op_desc, "_npu_sched_model", 1);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node0_ptr), SUCCESS);
  const auto &subgraphs = data_flow_graph.GetNodeSubgraphs();
  ASSERT_EQ(subgraphs.size(), 1);
  ASSERT_EQ(subgraphs.cbegin()->second.size(), 1);
  auto subgraph = subgraphs.cbegin()->second[0];
  int64_t value = 0;
  (void)AttrUtils::GetInt(subgraph, "_npu_sched_model", value);
  EXPECT_EQ(value, 1);
  std::string placement_value;
  (void)AttrUtils::GetStr(subgraph,ATTR_NAME_FLOW_ATTR_IO_PLACEMENT, placement_value);
  EXPECT_EQ(placement_value, "device");
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_balance_attr_check) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 2},
                                   {"output_num", 2},
                                   {"func_list", {{{"func_name", "func1"}}}},
                                   {"running_resources_info", {{{"type", "cpu"}, {"num", 2}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);

  DataFlowGraph data_flow_graph(root_graph, "", true);
  auto node0_ptr = root_graph->FindNode("node0");
  ASSERT_TRUE(node0_ptr != nullptr);
  auto op_desc = node0_ptr->GetOpDesc();
  ASSERT_TRUE(op_desc != nullptr);
  ASSERT_TRUE(AttrUtils::SetBool(op_desc, dflow::ATTR_NAME_BALANCE_GATHER, true));
  ASSERT_TRUE(AttrUtils::SetBool(op_desc, dflow::ATTR_NAME_BALANCE_SCATTER, true));
  EXPECT_NE(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node0_ptr), SUCCESS);

  ASSERT_TRUE(AttrUtils::SetBool(op_desc, dflow::ATTR_NAME_BALANCE_SCATTER, true));
  ASSERT_TRUE(AttrUtils::SetBool(op_desc, dflow::ATTR_NAME_BALANCE_GATHER, false));
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node0_ptr), SUCCESS);

  DataFlowGraph data_flow_graph1(root_graph, "", true);
  ASSERT_TRUE(AttrUtils::SetBool(op_desc, dflow::ATTR_NAME_BALANCE_SCATTER, false));
  ASSERT_TRUE(AttrUtils::SetBool(op_desc, dflow::ATTR_NAME_BALANCE_GATHER, true));
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph1, node0_ptr), SUCCESS);
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_With_InputShape_And_Name) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");

  remove("./pp1_config.json");
  std::string pp_config_file = "./pp1_config.json";
  {
      nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                      {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                       {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}},
                                       {"build_options", {{"ge.inputShape", "data0:1~3,2,4~10;data1:2~3,2~3,3"}}}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }

  DEF_GRAPH(graph_pp_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto node2_ptr = root_graph->FindNode("node0");
  EXPECT_TRUE(node2_ptr != nullptr);

  auto sub_graph_pp0 = ToComputeGraph(graph_pp_def);
  sub_graph_pp0->SetParentNode(node2_ptr);
  sub_graph_pp0->SetParentGraph(root_graph);
  sub_graph_pp0->SetName("graph_pp0");
  root_graph->AddSubgraph("graph_pp0", sub_graph_pp0);

  auto graph_pp0 = dataflow::ProcessPoint();
  graph_pp0.set_name("graph_pp0");
  graph_pp0.add_graphs("graph_pp0");
  graph_pp0.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  graph_pp0.set_compile_cfg_file(pp_config_file);
  EXPECT_EQ(AttrUtils::SetBool(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR, true), true);
  EXPECT_EQ(AttrUtils::SetInt(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR_DEPTH, 128), true);
  EXPECT_EQ(AttrUtils::SetStr(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, "FIFO"), true);
  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(graph_pp0, data_flow_graph, node2_ptr), SUCCESS);
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_With_InputShape_Without_Name) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");

  remove("./pp1_config.json");
  std::string pp_config_file = "./pp1_config.json";
  {
      nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                      {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                       {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}},
                                       {"build_options", {{"ge.inputShape", "1~3,2,4~10;2~3,2~3,3"}}}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }

  DEF_GRAPH(graph_pp_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto node2_ptr = root_graph->FindNode("node0");
  EXPECT_TRUE(node2_ptr != nullptr);

  auto sub_graph_pp0 = ToComputeGraph(graph_pp_def);
  sub_graph_pp0->SetParentNode(node2_ptr);
  sub_graph_pp0->SetParentGraph(root_graph);
  sub_graph_pp0->SetName("graph_pp0");
  root_graph->AddSubgraph("graph_pp0", sub_graph_pp0);

  auto graph_pp0 = dataflow::ProcessPoint();
  graph_pp0.set_name("graph_pp0");
  graph_pp0.add_graphs("graph_pp0");
  graph_pp0.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  graph_pp0.set_compile_cfg_file(pp_config_file);
  EXPECT_EQ(AttrUtils::SetBool(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR, true), true);
  EXPECT_EQ(AttrUtils::SetInt(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR_DEPTH, 128), true);
  EXPECT_EQ(AttrUtils::SetStr(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, "FIFO"), true);
  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(graph_pp0, data_flow_graph, node2_ptr), SUCCESS);
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_without_outputnode) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");

  remove("./pp1_config.json");
  std::string pp_config_file = "./pp1_config.json";
  {
    nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                    {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}}}},
                                   {"build_options", {{"ge.inputShape", "1~3,2,4~10"}}}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }

  DEF_GRAPH(graph_pp_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0);
    auto node0 = OP_CFG("Neg").InCnt(1).OutCnt(1);
    auto netoutput = OP_CFG("NetOutput").InCnt(1).OutCnt(1);
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->NODE("netoutput", netoutput));
  };
  auto node2_ptr = root_graph->FindNode("node0");
  EXPECT_TRUE(node2_ptr != nullptr);

  auto sub_graph_pp0 = ToComputeGraph(graph_pp_def);
  auto add_node = sub_graph_pp0->FindNode("node0");
  sub_graph_pp0->RemoveOutputNode(add_node);
  NodePtr output_node = sub_graph_pp0->FindFirstNodeMatchType("NetOutput");
  EXPECT_TRUE(output_node != nullptr);

  sub_graph_pp0->SetParentNode(node2_ptr);
  sub_graph_pp0->SetParentGraph(root_graph);
  sub_graph_pp0->SetName("graph_pp0");
  root_graph->AddSubgraph("graph_pp0", sub_graph_pp0);

  auto graph_pp0 = dataflow::ProcessPoint();
  graph_pp0.set_name("graph_pp0");
  graph_pp0.add_graphs("graph_pp0");
  graph_pp0.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  graph_pp0.set_compile_cfg_file(pp_config_file);
  EXPECT_EQ(AttrUtils::SetBool(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR, true), true);
  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(graph_pp0, data_flow_graph, node2_ptr), SUCCESS);
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_With_InputShape_Missing_Name) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");

  remove("./pp1_config.json");
  std::string pp_config_file = "./pp1_config.json";
  {
      nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                      {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                       {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}},
                                       {"build_options", {{"ge.inputShape", "data3:1~3,2,4~10;data1:2~3,2~3,3"}}}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }

  DEF_GRAPH(graph_pp_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto node2_ptr = root_graph->FindNode("node0");
  EXPECT_TRUE(node2_ptr != nullptr);

  auto sub_graph_pp0 = ToComputeGraph(graph_pp_def);
  sub_graph_pp0->SetParentNode(node2_ptr);
  sub_graph_pp0->SetParentGraph(root_graph);
  sub_graph_pp0->SetName("graph_pp0");
  root_graph->AddSubgraph("graph_pp0", sub_graph_pp0);

  auto graph_pp0 = dataflow::ProcessPoint();
  graph_pp0.set_name("graph_pp0");
  graph_pp0.add_graphs("graph_pp0");
  graph_pp0.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  graph_pp0.set_compile_cfg_file(pp_config_file);
  EXPECT_EQ(AttrUtils::SetBool(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR, true), true);
  EXPECT_EQ(AttrUtils::SetInt(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR_DEPTH, 128), true);
  EXPECT_EQ(AttrUtils::SetStr(node2_ptr->GetOpDesc(), ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, "FIFO"), true);
  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(graph_pp0, data_flow_graph, node2_ptr), FAILED);
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_Failed_invalid_config_json) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  // inputs_index out of range
  std::string pp0_error_config_file1 = "./pp0_error_config1.json";
  {
    nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 4},
                                   {"output_num", 2},
                                   {"func_list",
                                    {{{"func_name", "func1"}, {"inputs_index", {1, 2}}, {"outputs_index", {1}}},
                                     {{"func_name", "func2"}, {"inputs_index", {3, 4}}, {"outputs_index", {1}}}}}};
    std::ofstream json_file(pp0_error_config_file1);
    json_file << pp0_cfg_json << std::endl;
  }
  // outputs_index out of range
  std::string pp0_error_config_file2 = "./pp0_error_config2.json";
  {
    nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 4},
                                   {"output_num", 2},
                                   {"func_list",
                                    {{{"func_name", "func1"}, {"inputs_index", {0, 1}}, {"outputs_index", {1}}},
                                     {{"func_name", "func2"}, {"inputs_index", {2, 3}}, {"outputs_index", {2}}}}}};
    std::ofstream json_file(pp0_error_config_file2);
    json_file << pp0_cfg_json << std::endl;
  }
  // input index used by multi funcs
  std::string pp0_error_config_file3 = "./pp0_error_config3.json";
  {
    nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 4},
                                   {"output_num", 2},
                                   {"func_list",
                                    {{{"func_name", "func1"}, {"inputs_index", {1, 2}}, {"outputs_index", {1}}},
                                     {{"func_name", "func2"}, {"inputs_index", {1, 3}}, {"outputs_index", {1}}}}}};
    std::ofstream json_file(pp0_error_config_file3);
    json_file << pp0_cfg_json << std::endl;
  }
  // input num invalid
  std::string pp0_error_config_file4 = "./pp0_error_config4.json";
  {
    nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 5},
                                   {"output_num", 2},
                                   {"func_list",
                                    {{{"func_name", "func1"}, {"inputs_index", {0, 1}}, {"outputs_index", {0}}},
                                     {{"func_name", "func2"}, {"inputs_index", {2, 3}}, {"outputs_index", {1}}}}}};
    std::ofstream json_file(pp0_error_config_file4);
    json_file << pp0_cfg_json << std::endl;
  }
  // not specify inputs_index
  std::string pp0_error_config_file5 = "./pp0_error_config5.json";
  {
    nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 2},
                                   {"output_num", 2},
                                   {"func_list", {{{"func_name", "func1"}}, {{"func_name", "func2"}}}}};
    std::ofstream json_file(pp0_error_config_file5);
    json_file << pp0_cfg_json << std::endl;
  }
  // duplicate func names
  std::string pp0_error_config_file6 = "./pp0_error_config6.json";
  {
    nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 4},
                                   {"output_num", 2},
                                   {"func_list",
                                    {{{"func_name", "func1"}, {"inputs_index", {0, 1}}, {"outputs_index", {0}}},
                                     {{"func_name", "func1"}, {"inputs_index", {2, 3}}, {"outputs_index", {1}}}}}};
    std::ofstream json_file(pp0_error_config_file6);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_error_config_file1);

  DataFlowGraph data_flow_graph(root_graph);
  auto node_ptr = root_graph->FindNode("node0");
  EXPECT_TRUE(node_ptr != nullptr);
  EXPECT_NE(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node_ptr), SUCCESS);
  pp0.set_compile_cfg_file(pp0_error_config_file2);
  EXPECT_NE(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node_ptr), SUCCESS);
  pp0.set_compile_cfg_file(pp0_error_config_file3);
  EXPECT_NE(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node_ptr), SUCCESS);
  pp0.set_compile_cfg_file(pp0_error_config_file4);
  EXPECT_NE(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node_ptr), SUCCESS);
  pp0.set_compile_cfg_file(pp0_error_config_file5);
  EXPECT_NE(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node_ptr), SUCCESS);
  pp0.set_compile_cfg_file(pp0_error_config_file6);
  EXPECT_NE(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node_ptr), SUCCESS);
  remove(pp0_error_config_file1.c_str());
  remove(pp0_error_config_file2.c_str());
  remove(pp0_error_config_file3.c_str());
  remove(pp0_error_config_file4.c_str());
  remove(pp0_error_config_file5.c_str());
  remove(pp0_error_config_file6.c_str());
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_succ_buf_cfg) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
      nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
        {"target_bin", "libtest.so"},
        {"input_num", 2},
        {"output_num", 2},
        {"func_list", {{{"func_name", "func1"}}}},
        {"running_resources_info", {{{"type", "cpu"}, {"num", 2}}}},
        {"buf_cfg", 
        {
          {{"total_size", 2097152}, {"blk_size", 256}, {"max_buf_size", 8192}, {"page_type", "normal"}},
          {{"total_size", 33554432}, {"blk_size", 8192}, {"max_buf_size", 8388608}, {"page_type", "normal"}},
          {{"total_size", 2097152}, {"blk_size", 256}, {"max_buf_size", 8192}, {"page_type", "huge"}},
          {{"total_size", 33554432}, {"blk_size", 8192}, {"max_buf_size", 8388608}, {"page_type", "huge"}},
        }
      },
    };
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);

  DataFlowGraph data_flow_graph(root_graph, "", true);
  auto node0_ptr = root_graph->FindNode("node0");
  ASSERT_TRUE(node0_ptr != nullptr);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node0_ptr), SUCCESS);
  remove(pp0_config_file.c_str());
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_failed_buf_cfg_normal_disorder) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
      nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
        {"target_bin", "libtest.so"},
        {"input_num", 2},
        {"output_num", 2},
        {"func_list", {{{"func_name", "func1"}}}},
        {"running_resources_info", {{{"type", "cpu"}, {"num", 2}}}},
        {"buf_cfg", 
        {{{"total_size", 33554432}, {"blk_size", 8192}, {"max_buf_size", 8388608}, {"page_type", "normal"}},
          {{"total_size", 2097152}, {"blk_size", 256}, {"max_buf_size", 8192}, {"page_type", "normal"}},
          {{"total_size", 2097152}, {"blk_size", 256}, {"max_buf_size", 8192}, {"page_type", "huge"}},
          {{"total_size", 33554432}, {"blk_size", 8192}, {"max_buf_size", 8388608}, {"page_type", "huge"}},
        }
      },
    };
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);

  DataFlowGraph data_flow_graph(root_graph, "", true);
  auto node0_ptr = root_graph->FindNode("node0");
  ASSERT_TRUE(node0_ptr != nullptr);
  auto op_desc = node0_ptr->GetOpDesc();
  ASSERT_TRUE(op_desc != nullptr);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node0_ptr), FAILED);
  remove(pp0_config_file.c_str());
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_failed_buf_cfg_huge_disorder) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
      nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
        {"target_bin", "libtest.so"},
        {"input_num", 2},
        {"output_num", 2},
        {"func_list", {{{"func_name", "func1"}}}},
        {"running_resources_info", {{{"type", "cpu"}, {"num", 2}}}},
        {"buf_cfg", 
        {
          {{"total_size", 2097152}, {"blk_size", 256}, {"max_buf_size", 8192}, {"page_type", "normal"}},
          {{"total_size", 33554432}, {"blk_size", 8192}, {"max_buf_size", 8388608}, {"page_type", "normal"}},
          {{"total_size", 2097152}, {"blk_size", 256}, {"max_buf_size", 8192}, {"page_type", "huge"}},
          {{"total_size", 33554432}, {"blk_size", 256}, {"max_buf_size", 1024}, {"page_type", "huge"}},
        }
      },
    };
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);

  DataFlowGraph data_flow_graph(root_graph, "", true);
  auto node0_ptr = root_graph->FindNode("node0");
  ASSERT_TRUE(node0_ptr != nullptr);
  auto op_desc = node0_ptr->GetOpDesc();
  ASSERT_TRUE(op_desc != nullptr);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node0_ptr), FAILED);
  remove(pp0_config_file.c_str());
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_failed_buf_cfg_blk_invalid) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
      nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
        {"target_bin", "libtest.so"},
        {"input_num", 2},
        {"output_num", 2},
        {"func_list", {{{"func_name", "func1"}}}},
        {"running_resources_info", {{{"type", "cpu"}, {"num", 2}}}},
        {"buf_cfg", 
        {
          {{"total_size", 2097152}, {"blk_size", 255}, {"max_buf_size", 8192}, {"page_type", "normal"}},
        }
      },
    };
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);

  DataFlowGraph data_flow_graph(root_graph, "", true);
  auto node0_ptr = root_graph->FindNode("node0");
  ASSERT_TRUE(node0_ptr != nullptr);
  auto op_desc = node0_ptr->GetOpDesc();
  ASSERT_TRUE(op_desc != nullptr);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node0_ptr), PARAM_INVALID);
  remove(pp0_config_file.c_str());
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_failed_buf_cfg_page_type_invalid) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
      nlohmann::json pp0_cfg_json = {{"workspace", "temp"},
        {"target_bin", "libtest.so"},
        {"input_num", 2},
        {"output_num", 2},
        {"func_list", {{{"func_name", "func1"}}}},
        {"running_resources_info", {{{"type", "cpu"}, {"num", 2}}}},
        {"buf_cfg", 
        {
          {{"total_size", 2097152}, {"blk_size", 256}, {"max_buf_size", 8192}, {"page_type", "abnormal"}},
        }
      },
    };
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);

  DataFlowGraph data_flow_graph(root_graph, "", true);
  auto node0_ptr = root_graph->FindNode("node0");
  ASSERT_TRUE(node0_ptr != nullptr);
  auto op_desc = node0_ptr->GetOpDesc();
  ASSERT_TRUE(op_desc != nullptr);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(pp0, data_flow_graph, node0_ptr), FAILED);
  remove(pp0_config_file.c_str());
}

TEST_F(ProcessPointLoaderTest, LoadProcessPoint_Failed_invoke_key_repeat) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
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
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("Add", add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("Add", add));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  auto node_ptr = root_graph->FindNode("node0");
  EXPECT_TRUE(node_ptr != nullptr);

  auto sub_graph_pp0 = ToComputeGraph(pp0_graph);
  auto sub_graph_pp1 = ToComputeGraph(pp1_graph);
  sub_graph_pp0->SetParentNode(node_ptr);
  sub_graph_pp0->SetParentGraph(root_graph);
  sub_graph_pp0->SetName("pp0");
  root_graph->AddSubgraph("pp0", sub_graph_pp0);
  sub_graph_pp1->SetParentNode(node_ptr);
  sub_graph_pp1->SetParentGraph(root_graph);
  sub_graph_pp1->SetName("pp1");
  root_graph->AddSubgraph("pp1", sub_graph_pp1);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  auto pp1 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp1.set_name("pp1");
  pp0.add_graphs("pp0");
  pp1.add_graphs("pp1");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);

  auto func_pp0 = dataflow::ProcessPoint();
  auto func_pp1 = dataflow::ProcessPoint();
  func_pp0.set_name("func_pp0");
  func_pp1.set_name("func_pp1");
  func_pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  func_pp1.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string func_config_file = "./func_config.json";
  {
    nlohmann::json func_pp_cfg_json = {{"workspace", "temp"},
                                       {"target_bin", "libtest.so"},
                                       {"input_num", 2},
                                       {"output_num", 2},
                                       {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(func_config_file);
    json_file << func_pp_cfg_json << std::endl;
  }
  func_pp0.set_compile_cfg_file(func_config_file);
  func_pp1.set_compile_cfg_file(func_config_file);

  auto func_pp0_invoke_pps = func_pp0.mutable_invoke_pps();
  (*func_pp0_invoke_pps)["invoke_key"] = pp0;
  auto func_pp1_invoke_pps = func_pp1.mutable_invoke_pps();
  (*func_pp1_invoke_pps)["invoke_key"] = pp1;

  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_EQ(ProcessPointLoader::LoadProcessPoint(func_pp0, data_flow_graph, node_ptr), SUCCESS);
  EXPECT_NE(ProcessPointLoader::LoadProcessPoint(func_pp1, data_flow_graph, node_ptr), SUCCESS);
  remove(func_config_file.c_str());
}
}  // namespace ge
