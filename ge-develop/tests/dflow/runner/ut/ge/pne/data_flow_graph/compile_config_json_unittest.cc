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
#include "dflow/compiler/data_flow_graph/compile_config_json.h"
#include "mmpa/mmpa_api.h"
#include "graph/ge_global_options.h"
#include "framework/common/ge_types.h"

using namespace testing;
namespace ge {
class CompileConfigJsonTest : public Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(CompileConfigJsonTest, ReadFunctionPpConfigFromJsonFile_InvalidJsonFile) {
  CompileConfigJson::FunctionPpConfig func_pp_cfg = {};
  EXPECT_EQ(CompileConfigJson::ReadFunctionPpConfigFromJsonFile("./func_pp_config.json", func_pp_cfg), FAILED);
}

TEST_F(CompileConfigJsonTest, ReadFunctionPpConfigFromJsonFile_InvalidIputsIndex) {
  nlohmann::json func_pp_cfg_json = {
      {"workspace", "xxx_path"},
      {"target_bin", "libxxx.so"},
      {"input_num", 2},
      {"output_num", 2},
      {"func_list", {{{"func_name", "func1"}, {"inputs_index", {"1"}}, {"outputs_index", {"1"}}}}}};
  std::ofstream json_file("./func_pp_config.json");
  json_file << func_pp_cfg_json << std::endl;
  CompileConfigJson::FunctionPpConfig func_pp_cfg = {};
  EXPECT_EQ(CompileConfigJson::ReadFunctionPpConfigFromJsonFile("./func_pp_config.json", func_pp_cfg), FAILED);
  remove("./func_pp_config.json");
}

TEST_F(CompileConfigJsonTest, ReadFunctionPpConfigFromJsonFile_stream_input) {
  nlohmann::json func_pp_cfg_json = {
      {"workspace", "xxx_path"},
      {"target_bin", "libxxx.so"},
      {"input_num", 2},
      {"output_num", 2},
      {"func_list", {{{"func_name", "func1"}, {"stream_input", true}}}},
      {"cmakelist_path", "CMakeLists.txt"},
  };
  std::ofstream json_file("./func_pp_config.json");
  json_file << func_pp_cfg_json << std::endl;
  CompileConfigJson::FunctionPpConfig func_pp_cfg = {};
  EXPECT_EQ(CompileConfigJson::ReadFunctionPpConfigFromJsonFile("./func_pp_config.json", func_pp_cfg), SUCCESS);
  EXPECT_EQ(func_pp_cfg.funcs.size(), 1);
  EXPECT_TRUE(func_pp_cfg.funcs[0].stream_input);
  remove("./func_pp_config.json");
}

TEST_F(CompileConfigJsonTest, ReadFunctionPpConfigFromJsonFile_compile) {
  nlohmann::json cpu_compiler_json = {
      {"compiler",
       {
           {
               {"resource_type", "X86"},
               {"toolchain", "/usr/bin/g++"},
           },
           {
               {"resource_type", "Aarch64"},
               {"toolchain", "/usr/bin/g++"},
           },
           {
               {"resource_type", "Ascend"},
               {"toolchain", "/usr/local/Ascend/hcc"},
           },
       }},
  };
  std::ofstream cpu_compiler_file("./cpu_compile.json");
  cpu_compiler_file << cpu_compiler_json << std::endl;
  nlohmann::json func_pp_cfg_json = {
      {"workspace", "xxx_path"},
      {"target_bin", "libxxx.so"},
      {"input_num", 2},
      {"output_num", 2},
      {"func_list", {{{"func_name", "func1"}}}},
      {"cmakelist_path", "CMakeLists.txt"},
      {"compiler", "cpu_compile.json"},
      {
          "running_resources_info",
          {
              {
                  {"type", "cpu"},
                  {"num", 2},
              },
              {
                  {"type", "memory"},
                  {"num", 100},
              },
          },
      },
  };
  std::ofstream json_file("./func_pp_config.json");
  json_file << func_pp_cfg_json << std::endl;
  CompileConfigJson::FunctionPpConfig func_pp_cfg = {};
  EXPECT_EQ(CompileConfigJson::ReadFunctionPpConfigFromJsonFile("./func_pp_config.json", func_pp_cfg), SUCCESS);
  EXPECT_EQ(func_pp_cfg.cmakelist, "CMakeLists.txt");
  EXPECT_FALSE(func_pp_cfg.heavy_load);
  EXPECT_EQ(func_pp_cfg.toolchain_map.size(), 3);
  EXPECT_EQ(func_pp_cfg.running_resources_info.size(), 2);
  EXPECT_EQ(func_pp_cfg.running_resources_info[0].resource_type, "cpu");
  EXPECT_EQ(func_pp_cfg.running_resources_info[0].resource_num, 2);
  EXPECT_EQ(func_pp_cfg.running_resources_info[1].resource_type, "memory");
  EXPECT_EQ(func_pp_cfg.running_resources_info[1].resource_num, 100);
  remove("./func_pp_config.json");
  remove("./cpu_compile.json");
}

TEST_F(CompileConfigJsonTest, ReadFunctionPpConfigFromJsonFile_heavy_load) {
  nlohmann::json func_pp_cfg_json = {
      {"workspace", "xxx_path"},
      {"target_bin", "libxxx.so"},
      {"input_num", 2},
      {"output_num", 2},
      {"heavy_load", true},
      {"visible_device_enable", true},
      {"func_list", {{{"func_name", "func1"}}}},
      {"cmakelist_path", "CMakeLists.txt"},
  };
  std::ofstream json_file("./func_pp_config.json");
  json_file << func_pp_cfg_json << std::endl;
  CompileConfigJson::FunctionPpConfig func_pp_cfg = {};
  EXPECT_EQ(CompileConfigJson::ReadFunctionPpConfigFromJsonFile("./func_pp_config.json", func_pp_cfg), SUCCESS);
  EXPECT_TRUE(func_pp_cfg.heavy_load);
  EXPECT_TRUE(func_pp_cfg.visible_device_enable);
  remove("./func_pp_config.json");
}

TEST_F(CompileConfigJsonTest, ReadFunctionPpConfigFromJsonFile_buf_cfg) {
  nlohmann::json func_pp_cfg_json = {
      {"workspace", "xxx_path"},
      {"target_bin", "libxxx.so"},
      {"input_num", 2},
      {"output_num", 2},
      {"heavy_load", true},
      {"func_list", {{{"func_name", "func1"}}}},
      {"cmakelist_path", "CMakeLists.txt"},
      {"buf_cfg", 
      {
        {{"total_size", 2097152}, {"blk_size", 256}, {"max_buf_size", 8192}, {"page_type", "normal"}},
        {{"total_size", 33554432}, {"blk_size", 8192}, {"max_buf_size", 8388608}, {"page_type", "normal"}},
        {{"total_size", 2097152}, {"blk_size", 256}, {"max_buf_size", 8192}, {"page_type", "huge"}},
        {{"total_size", 33554432}, {"blk_size", 8192}, {"max_buf_size", 8388608}, {"page_type", "huge"}},
      }
    },
  };
  std::ofstream json_file("./func_pp_config.json");
  json_file << func_pp_cfg_json << std::endl;
  CompileConfigJson::FunctionPpConfig func_pp_cfg = {};
  EXPECT_EQ(CompileConfigJson::ReadFunctionPpConfigFromJsonFile("./func_pp_config.json", func_pp_cfg), SUCCESS);
  EXPECT_TRUE(func_pp_cfg.heavy_load);
  EXPECT_EQ(func_pp_cfg.user_buf_cfg.size(), 4);
  EXPECT_EQ(func_pp_cfg.user_buf_cfg[0].total_size, 2097152);
  EXPECT_EQ(func_pp_cfg.user_buf_cfg[1].blk_size, 8192);
  EXPECT_EQ(func_pp_cfg.user_buf_cfg[2].max_buf_size, 8192);
  EXPECT_EQ(func_pp_cfg.user_buf_cfg[3].page_type, "huge");
  remove("./func_pp_config.json");
}

TEST_F(CompileConfigJsonTest, ReadGraphPpConfigFromJsonFile_InvalidInputsTensorDescShape) {
  nlohmann::json graph_pp_cfg_json = {{"build_options", {{"key1", "value1"}}},
                                      {"inputs_tensor_desc", {{{"shape", {"1", "2"}}, {"data_type", "DT_UINT32"}}}}};
  std::ofstream json_file("./graph_pp_config.json");
  json_file << graph_pp_cfg_json << std::endl;
  CompileConfigJson::GraphPpConfig graph_pp_cfg = {};
  EXPECT_EQ(CompileConfigJson::ReadGraphPpConfigFromJsonFile("./graph_pp_config.json", graph_pp_cfg), FAILED);
  remove("./graph_pp_config.json");
}


TEST_F(CompileConfigJsonTest, ReadModelPpConfigFromJsonFileWithoutFile) {
  CompileConfigJson::ModelPpConfig  model_pp_config = {};
  EXPECT_EQ(CompileConfigJson::ReadModelPpConfigFromJsonFile("", model_pp_config), SUCCESS);
}

TEST_F(CompileConfigJsonTest, ReadDeployInfoFromJsonFile) {
  constexpr const char *file_name = "./deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node1",
            "logic_device_id":"0:1:0,0:1:1"
          },
          {
            "flow_node_name":"node2",
            "logic_device_id":"0:0:-1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  CompileConfigJson::DeployConfigInfo deploy_config;
  EXPECT_EQ(CompileConfigJson::ReadDeployInfoFromJsonFile(file_name, deploy_config), SUCCESS);
  remove(file_name);
  EXPECT_EQ(deploy_config.deploy_info_list.size(), 2);
  EXPECT_TRUE(deploy_config.batch_deploy_info_list.empty());
  for (const auto &deploy_info:deploy_config.deploy_info_list) {
    if (deploy_info.flow_node_name == "node1") {
      EXPECT_EQ(deploy_info.logic_device_id, "0:1:0,0:1:1");
    } else if (deploy_info.flow_node_name == "node2") {
      EXPECT_EQ(deploy_info.logic_device_id, "0:0:-1");
    } else {
      FAIL();
    }
  }
}

TEST_F(CompileConfigJsonTest, ReadDeployInfoFromJsonFile_batch_config) {
  constexpr const char *file_name = "./deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "dynamic_schedule_enable": true,
        "batch_deploy_info": [
          {
            "flow_node_list":["node1", "node2"],
            "logic_device_list":"0:0:1,0:1:0~1"
          },
          {
            "flow_node_list":["node3"],
            "logic_device_list":"0:0:-1"
          },
          {
            "flow_node_list":["node4", "node5"],
            "logic_device_list":"0:1~4:0~1",
            "redundant_logic_device_list":"0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  CompileConfigJson::DeployConfigInfo deploy_config;
  EXPECT_EQ(CompileConfigJson::ReadDeployInfoFromJsonFile(file_name, deploy_config), SUCCESS);
  remove(file_name);
  EXPECT_TRUE(deploy_config.deploy_info_list.empty());
  EXPECT_EQ(deploy_config.batch_deploy_info_list.size(), 3);
  EXPECT_EQ(deploy_config.dynamic_schedule_enable, true);
  std::map<std::string, std::pair<std::string, std::string>> deploy_map;
  for (const auto &batch_deploy_info : deploy_config.batch_deploy_info_list) {
    for (const auto &flow_node : batch_deploy_info.flow_node_list) {
      deploy_map.emplace(flow_node,
        std::make_pair(batch_deploy_info.logic_device_list,
                       batch_deploy_info.redundant_logic_device_list));
    }
  }
  EXPECT_EQ(deploy_map.size(), 5);
  EXPECT_EQ(deploy_map["node1"].first, "0:0:1,0:1:0~1");
  EXPECT_EQ(deploy_map["node2"].first, "0:0:1,0:1:0~1");
  EXPECT_EQ(deploy_map["node3"].first, "0:0:-1");
  EXPECT_EQ(deploy_map["node4"].first, "0:1~4:0~1");
  EXPECT_EQ(deploy_map["node5"].first, "0:1~4:0~1");

  EXPECT_EQ(deploy_map["node1"].second, "");
  EXPECT_EQ(deploy_map["node2"].second, "");
  EXPECT_EQ(deploy_map["node3"].second, "");
  EXPECT_EQ(deploy_map["node4"].second, "0:0:0");
  EXPECT_EQ(deploy_map["node5"].second, "0:0:0");
}

TEST_F(CompileConfigJsonTest, ReadDeployInfoFromJsonFile_both_config) {
  constexpr const char *file_name = "./deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node1",
            "logic_device_id":"0:0:1:0,0:0:1:1"
          },
          {
            "flow_node_name":"node2",
            "logic_device_id":"0:0:0:-1"
          }
        ],
        "dynamic_schedule_enable": false,
        "batch_deploy_info": [
          {
            "flow_node_list":["node11", "node12"],
            "logic_device_list":"0:0:0:1,0:0:1:0~1"
          },
          {
            "flow_node_list":["node13"],
            "logic_device_list":"0:0:0:-1"
          },
          {
            "flow_node_list":["node14", "node15"],
            "logic_device_list":"0:0:1~4:0~1"
          }
        ],
        "deploy_mem_info":[
          {
            "std_mem_size":"1024",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:0:1"
          },
{
            "std_mem_size":"1024",
            "shared_mem_size":"2048",
            "logic_device_id":"0:0:1~4:0~1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  CompileConfigJson::DeployConfigInfo deploy_config;
  EXPECT_EQ(CompileConfigJson::ReadDeployInfoFromJsonFile(file_name, deploy_config), SUCCESS);
  remove(file_name);
  EXPECT_EQ(deploy_config.deploy_info_list.size(), 2);
  EXPECT_EQ(deploy_config.batch_deploy_info_list.size(), 3);
  EXPECT_EQ(deploy_config.dynamic_schedule_enable, false);
  for (const auto &deploy_info:deploy_config.deploy_info_list) {
    if (deploy_info.flow_node_name == "node1") {
      EXPECT_EQ(deploy_info.logic_device_id, "0:0:1:0,0:0:1:1");
    } else if (deploy_info.flow_node_name == "node2") {
      EXPECT_EQ(deploy_info.logic_device_id, "0:0:0:-1");
    } else {
      FAIL();
    }
  }
  std::map<std::string, std::string> deploy_map;
  for (const auto &batch_deploy_info : deploy_config.batch_deploy_info_list) {
    for (const auto &flow_node : batch_deploy_info.flow_node_list) {
      deploy_map.emplace(flow_node, batch_deploy_info.logic_device_list);
    }
  }
  EXPECT_EQ(deploy_config.mem_size_cfg.size(), 2);
  EXPECT_EQ(deploy_map.size(), 5);
  EXPECT_EQ(deploy_map["node11"], "0:0:0:1,0:0:1:0~1");
  EXPECT_EQ(deploy_map["node12"], "0:0:0:1,0:0:1:0~1");
  EXPECT_EQ(deploy_map["node13"], "0:0:0:-1");
  EXPECT_EQ(deploy_map["node14"], "0:0:1~4:0~1");
  EXPECT_EQ(deploy_map["node15"], "0:0:1~4:0~1");
}

TEST_F(CompileConfigJsonTest, ReadDeployInfoFromJsonFile_no_config) {
  constexpr const char *file_name = "./deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"({})";
  json_file << content << std::endl;
  json_file.close();
  CompileConfigJson::DeployConfigInfo deploy_config;
  EXPECT_NE(CompileConfigJson::ReadDeployInfoFromJsonFile(file_name, deploy_config), SUCCESS);
  remove(file_name);
}

TEST_F(CompileConfigJsonTest, ReadDeployInfoFromJsonFile_json_invalid) {
  constexpr const char *file_name = "./deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node1",
            "logic_device_id":"1"
          },
          {
            "flow_node_name":"node2",
            ----json format error--- "logic_device_id":"0:0:-1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  CompileConfigJson::DeployConfigInfo deploy_config;
  EXPECT_NE(CompileConfigJson::ReadDeployInfoFromJsonFile(file_name, deploy_config), SUCCESS);
  remove(file_name);
}

TEST_F(CompileConfigJsonTest, ReadDeployInfoFromJsonFile_config_failed) {
  constexpr const char *file_name = "./deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node1",
            "logic_device_id":1
          },
          {
            "flow_node_name":"node2",
            "logic_device_id":"0:0:-1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  CompileConfigJson::DeployConfigInfo deploy_config;
  EXPECT_NE(CompileConfigJson::ReadDeployInfoFromJsonFile(file_name, deploy_config), SUCCESS);
  remove(file_name);
}

TEST_F(CompileConfigJsonTest, ReadGraphPpConfigFromJsonFile_NotSetJsonFile) {
  CompileConfigJson::GraphPpConfig graph_pp_cfg = {};
  EXPECT_EQ(CompileConfigJson::ReadGraphPpConfigFromJsonFile("", graph_pp_cfg), SUCCESS);
}

TEST_F(CompileConfigJsonTest, ReadGraphPpConfigFromJsonFile_SUCCESS) {
  nlohmann::json graph_pp_cfg_json = {
      {"build_options", {{"key1", "value1"}}},
      {"inputs_tensor_desc", {{{"shape", {1, 2}}, {"data_type", "DT_UINT32"}, {"format", "ND"}}}}};
  std::ofstream json_file("./graph_pp_config.json");
  json_file << graph_pp_cfg_json << std::endl;
  CompileConfigJson::GraphPpConfig graph_pp_cfg = {};
  EXPECT_EQ(CompileConfigJson::ReadGraphPpConfigFromJsonFile("./graph_pp_config.json", graph_pp_cfg), SUCCESS);
  EXPECT_EQ(graph_pp_cfg.inputs_tensor_desc_config.size(), 1);
  EXPECT_EQ(std::get<1>(graph_pp_cfg.inputs_tensor_desc_config[0].dtype_config), DT_UINT32);
  EXPECT_EQ(std::get<1>(graph_pp_cfg.inputs_tensor_desc_config[0].format_config), FORMAT_ND);
  remove("./graph_pp_config.json");
}

TEST_F(CompileConfigJsonTest, ReadGraphPpConfigFromJsonFile_InvalidDataType) {
  nlohmann::json graph_pp_cfg_json = {
      {"build_options", {{"key1", "value1"}}},
      {"inputs_tensor_desc", {{{"shape", {1, 2}}, {"data_type", "xxxx"}, {"format", "ND"}}}}};
  std::ofstream json_file("./graph_pp_config.json");
  json_file << graph_pp_cfg_json << std::endl;
  CompileConfigJson::GraphPpConfig graph_pp_cfg = {};
  EXPECT_EQ(CompileConfigJson::ReadGraphPpConfigFromJsonFile("./graph_pp_config.json", graph_pp_cfg), SUCCESS);
  EXPECT_EQ(graph_pp_cfg.inputs_tensor_desc_config.size(), 1);
  EXPECT_EQ(std::get<1>(graph_pp_cfg.inputs_tensor_desc_config[0].dtype_config), DT_UNDEFINED);
  remove("./graph_pp_config.json");
}

TEST_F(CompileConfigJsonTest, ReadGraphPpConfigFromJsonFile_InvalidFormat) {
  nlohmann::json graph_pp_cfg_json = {
      {"build_options", {{"key1", "value1"}}},
      {"inputs_tensor_desc", {{{"shape", {1, 2}}, {"data_type", "DT_UINT32"}, {"format", 1}}}}};
  std::ofstream json_file("./graph_pp_config.json");
  json_file << graph_pp_cfg_json << std::endl;
  CompileConfigJson::GraphPpConfig graph_pp_cfg = {};
  EXPECT_EQ(CompileConfigJson::ReadGraphPpConfigFromJsonFile("./graph_pp_config.json", graph_pp_cfg), FAILED);
  remove("./graph_pp_config.json");
}

TEST_F(CompileConfigJsonTest, GetResourceTypeFromNumaConfig_Failed) {

  {
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    auto &global_options = GetMutableGlobalOptions();
    global_options[OPTION_NUMA_CONFIG] = "xxxxx";
  }
  std::set<std::string> resource_types;
  EXPECT_EQ(CompileConfigJson::GetResourceTypeFromNumaConfig(resource_types), FAILED);
}
}  // namespace ge
