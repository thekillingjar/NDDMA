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
#include "proto/dflow.pb.h"
#include "nlohmann/json.hpp"
#include "dflow/inc/data_flow/flow_graph/model_pp.h"
#include "flow_graph/data_flow.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "dflow/compiler/data_flow_graph/inner_pp_loader.h"
#include "graph/ge_global_options.h"
#include "common/env_path.h"

using namespace testing;
namespace ge {
namespace {
void PrepareForUdf() {
  std::string cmd = "mkdir -p model_pp_udf; cd model_pp_udf; touch libtest.so";
  (void)system(cmd.c_str());
  std::ofstream cmakefile("./model_pp_udf/CMakeLists.txt");
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

  std::string model_pp_udf_config_file = "./model_pp_udf/model_pp_udf_config_file.json";
  {
    nlohmann::json model_pp_udf_cfg_json = {{"workspace", "./model_pp_udf"},
                                            {"target_bin", "libtest.so"},
                                            {"input_num", 1},
                                            {"output_num", 1},
                                            {"cmakelist_path", "CMakeLists.txt"},
                                            {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(model_pp_udf_config_file);
    json_file << model_pp_udf_cfg_json << std::endl;
  }
}
Graph BuildGraphWithInvokeModelPp(const std::string &name, const std::string &model_path) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto node0 = dflow::FlowNode("node0", 1, 1).SetInput(0, data0);

  auto invoked_model_pp0 = dflow::ModelPp("invoked_graph_pp0", model_path.c_str());
  // function pp
  auto udf_pp = dflow::FunctionPp("udf_pp")
                    .SetCompileConfig("./model_pp_udf/model_pp_udf_config_file.json")
                    .AddInvokedClosure("invoke_model_pp0", invoked_model_pp0);
  node0.AddPp(udf_pp);

  std::vector<dflow::FlowOperator> inputsOperator{data0};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph(name.c_str());
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph.ToGeGraph();
}

Graph BuildGraphWithInvokeModelPpWithConfigFile(const std::string &name, const std::string &model_path, int32_t type = 0) {
  // type 0: correct ;1 invalid path ;2 invalid json
  auto data0 = dflow::FlowData("Data0", 0);
  auto node0 = dflow::FlowNode("node0", 1, 1).SetInput(0, data0);
  std::string pp_config_file = "./config.json";
  {
    nlohmann::json pp1_cfg_json = {{"invoke_model_fusion_inputs", "0~7"}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  std::string pp_config_file1 = "./config1.json";
  {
    nlohmann::json pp1_cfg_json = {{"invoke_model_fusion_inputs", 10}};
    std::ofstream json_file(pp_config_file1);
    json_file << pp1_cfg_json << std::endl;
  }
  auto invoked_model_pp0 = dflow::ModelPp("invoked_graph_pp0", model_path.c_str());
  if (type == 0) {
    invoked_model_pp0.SetCompileConfig(pp_config_file.c_str());
  } else if (type == 1) {
    invoked_model_pp0.SetCompileConfig("error");
  } else {
    invoked_model_pp0.SetCompileConfig(pp_config_file1.c_str());
  }

  // function pp
  auto udf_pp = dflow::FunctionPp("udf_pp")
                    .SetCompileConfig("./model_pp_udf/model_pp_udf_config_file.json")
                    .AddInvokedClosure("invoke_model_pp0", invoked_model_pp0);
  node0.AddPp(udf_pp);

  std::vector<dflow::FlowOperator> inputsOperator{data0};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph(name.c_str());
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph.ToGeGraph();
}
}  // namespace
class InnerPpLoaderTest : public Test {
 protected:
  static void SetUpTestSuite() {
    PrepareForUdf();
    {
      auto &global_options_mutex = GetGlobalOptionsMutex();
      const std::lock_guard<std::mutex> lock(global_options_mutex);
      auto &global_options = GetMutableGlobalOptions();
      global_options[OPTION_NUMA_CONFIG] =
          R"({"cluster":[{"cluster_nodes":[{"is_local":true, "item_list":[{"item_id":0}], "node_id":0, "node_type":"TestNodeType1"}]}],"item_def":[{"aic_type":"[DAVINCI_V100:10]","item_type":"","memory":"[DDR:80GB]","resource_type":"Ascend"}],"node_def":[{"item_type":"","links_mode":"TCP:128Gb","node_type":"TestNodeType1","resource_type":"X86","support_links":"[ROCE]"}]})";
    }
  }
  static void TearDownTestSuite() {
    system("rm -fr model_pp_udf");
    {
      auto &global_options_mutex = GetGlobalOptionsMutex();
      const std::lock_guard<std::mutex> lock(global_options_mutex);
      auto &global_options = GetMutableGlobalOptions();
      global_options[OPTION_NUMA_CONFIG] = "";
    }
  }
  void SetUp() {}
  void TearDown() {}
  const std::string run_data_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/ge/st/st_run_data/"});
};
TEST_F(InnerPpLoaderTest, model_pp_load) {
  Graph graph = BuildGraphWithInvokeModelPp("testModelPp", run_data_path + "origin_model/root_model.om");
  const auto &comput_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void)AttrUtils::SetStr(comput_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  DataFlowGraph data_flow_graph(comput_graph);
  ASSERT_EQ(data_flow_graph.Initialize(), SUCCESS);
  const auto &all_loaded_models = data_flow_graph.GetAllLoadedModels();
  ASSERT_TRUE(all_loaded_models.find("invoked_graph_pp0") != all_loaded_models.cend());
  const auto &node_loaded_models = data_flow_graph.GetNodeLoadedModels();
  ASSERT_FALSE(node_loaded_models.empty());
  ASSERT_TRUE(node_loaded_models.find("node0") != node_loaded_models.cend());
  std::vector<std::string> expect_invoke_keys = {"invoke_model_pp0"};
  ASSERT_EQ(data_flow_graph.GetInvokeKeys("udf_pp"), expect_invoke_keys);
}

TEST_F(InnerPpLoaderTest, model_pp_load_error_config_file) {
  Graph graph = BuildGraphWithInvokeModelPpWithConfigFile("testModelPp", run_data_path + "origin_model/root_model.om", 1);
  const auto &comput_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void)AttrUtils::SetStr(comput_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  DataFlowGraph data_flow_graph(comput_graph);
  ASSERT_EQ(data_flow_graph.Initialize(), FAILED);
  (void)system("rm -rf ./config.json; rm -rf ./config1.json");
}

TEST_F(InnerPpLoaderTest, model_pp_load_error_config_format) {
  Graph graph = BuildGraphWithInvokeModelPpWithConfigFile("testModelPp", run_data_path + "origin_model/root_model.om", 2);
  const auto &comput_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void)AttrUtils::SetStr(comput_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  DataFlowGraph data_flow_graph(comput_graph);
  ASSERT_EQ(data_flow_graph.Initialize(), FAILED);
  (void)system("rm -rf ./config.json; rm -rf ./config1.json");
}

TEST_F(InnerPpLoaderTest, model_pp_load_correct_config_file) {
  Graph graph = BuildGraphWithInvokeModelPpWithConfigFile("testModelPp", run_data_path + "origin_model/root_model.om", 0);
  const auto &comput_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void)AttrUtils::SetStr(comput_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  DataFlowGraph data_flow_graph(comput_graph);
  ASSERT_EQ(data_flow_graph.Initialize(), SUCCESS);
  (void)system("rm -rf ./config.json; rm -rf ./config1.json");
}

TEST_F(InnerPpLoaderTest, model_pp_load_model_not_exist) {
  Graph graph = BuildGraphWithInvokeModelPp("testModelPp", "./not_exist_model.om");
  const auto &comput_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void)AttrUtils::SetStr(comput_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  DataFlowGraph data_flow_graph(comput_graph);
  ASSERT_NE(data_flow_graph.Initialize(), SUCCESS);
}

TEST_F(InnerPpLoaderTest, load_custom_inner_type_not_exist) {
  Graph graph = BuildGraphWithInvokeModelPp("testModelPp", "./test_model.om");
  const auto &comput_graph = GraphUtilsEx::GetComputeGraph(graph);
  DataFlowGraph data_flow_graph(comput_graph);

  const auto &all_nodes = comput_graph->GetAllNodes();
  ASSERT_FALSE(all_nodes.empty());
  NodePtr node = *(all_nodes.begin());
  dataflow::ProcessPoint process_point;
  process_point.set_name("test_name");
  process_point.set_type(dataflow::ProcessPoint_ProcessPointType_INNER);
  EXPECT_NE(InnerPpLoader::LoadProcessPoint(process_point, data_flow_graph, node), SUCCESS);
}

TEST_F(InnerPpLoaderTest, load_unknown_inner_type) {
  Graph graph = BuildGraphWithInvokeModelPp("testModelPp", "./test_model.om");
  const auto &comput_graph = GraphUtilsEx::GetComputeGraph(graph);
  DataFlowGraph data_flow_graph(comput_graph);

  const auto &all_nodes = comput_graph->GetAllNodes();
  ASSERT_FALSE(all_nodes.empty());
  NodePtr node = *(all_nodes.begin());
  dataflow::ProcessPoint process_point;
  process_point.set_name("test_name");
  process_point.set_type(dataflow::ProcessPoint_ProcessPointType_INNER);
  auto *const pp_extend_attrs = process_point.mutable_pp_extend_attrs();
  (*pp_extend_attrs)[std::string(dflow::INNER_PP_CUSTOM_ATTR_INNER_TYPE)] = "unkown_inner_type";

  EXPECT_NE(InnerPpLoader::LoadProcessPoint(process_point, data_flow_graph, node), SUCCESS);
}

TEST_F(InnerPpLoaderTest, load_model_pp_model_path_attr_not_exist) {
  Graph graph = BuildGraphWithInvokeModelPp("testModelPp", "./test_model.om");
  const auto &comput_graph = GraphUtilsEx::GetComputeGraph(graph);
  DataFlowGraph data_flow_graph(comput_graph);

  const auto &all_nodes = comput_graph->GetAllNodes();
  ASSERT_FALSE(all_nodes.empty());
  NodePtr node = *(all_nodes.begin());
  dataflow::ProcessPoint process_point;
  process_point.set_name("test_name");
  process_point.set_type(dataflow::ProcessPoint_ProcessPointType_INNER);
  auto *const pp_extend_attrs = process_point.mutable_pp_extend_attrs();
  (*pp_extend_attrs)[std::string(dflow::INNER_PP_CUSTOM_ATTR_INNER_TYPE)] = dflow::INNER_PP_TYPE_MODEL_PP;

  EXPECT_NE(InnerPpLoader::LoadProcessPoint(process_point, data_flow_graph, node), SUCCESS);
}
}  // namespace ge
