/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_PNE_DATA_FLOW_GRAPH_COMPILE_CONFIG_JSON_H
#define AIR_COMPILER_PNE_DATA_FLOW_GRAPH_COMPILE_CONFIG_JSON_H

#include <map>
#include <string>
#include <vector>
#include "framework/common/debug/log.h"
#include "graph/ge_tensor.h"
#include "nlohmann/json.hpp"

namespace ge {
constexpr const char *kResourceTypeAscend = "Ascend";
class CompileConfigJson {
 public:
  struct FunctionDesc {
    std::string func_name;
    std::vector<uint32_t> inputs_index;
    std::vector<uint32_t> outputs_index;
    bool stream_input = false;
  };

  struct RunningResourceInfo {
    std::string resource_type;
    uint32_t resource_num;
  };

  struct BufCfg {
    uint32_t total_size;   // total pool size
    uint32_t blk_size;     // min size alloced at once
    uint32_t max_buf_size; // max size alloced at once
    std::string page_type; // page type:normal or huge
  };

  struct FunctionPpConfig {
    std::string workspace;
    std::string target_bin;
    uint32_t input_num;
    uint32_t output_num;
    std::vector<FunctionDesc> funcs;
    std::string cmakelist;
    std::map<std::string, std::string> toolchain_map;
    std::vector<RunningResourceInfo> running_resources_info;
    bool heavy_load = false;
    bool built_in_flow_func = false;
    std::vector<BufCfg> user_buf_cfg;
    bool visible_device_enable = false;
  };

  struct TensorDescConfig {
    std::tuple<bool, std::vector<int64_t>> shape_config;
    std::tuple<bool, DataType> dtype_config;
    std::tuple<bool, Format> format_config;
  };

  struct GraphPpConfig {
    std::map<std::string, std::string> build_options;
    std::vector<TensorDescConfig> inputs_tensor_desc_config;
  };

  struct ModelPpConfig {
    std::string invoke_model_fusion_inputs;
  };

  struct FlowNodeDeployInfo {
    std::string flow_node_name;
    std::string logic_device_id;
  };

  struct InvokeDeployInfo {
    std::string invoke_name;
    std::string deploy_info_file;
    std::string logic_device_list;
    std::string redundant_logic_device_list;
  };

  struct FlowNodeBatchDeployInfo {
    std::vector<std::string> flow_node_list;
    std::string logic_device_list;
    std::string redundant_logic_device_list;
    std::vector<InvokeDeployInfo> invoke_deploy_infos;
  };

  struct FlowNodeBatchMemCfg {
    std::string std_mem_size;
    std::string shared_mem_size;
    std::string logic_device_list;
  };

  struct DeployConfigInfo {
    std::vector<FlowNodeDeployInfo> deploy_info_list;
    std::vector<FlowNodeBatchDeployInfo> batch_deploy_info_list;
    std::vector<FlowNodeBatchMemCfg> mem_size_cfg;
    bool dynamic_schedule_enable = false;
    bool keep_logic_device_order = false;
  };

  static Status ReadFunctionPpConfigFromJsonFile(const std::string &file_path, FunctionPpConfig &function_pp_config);
  static Status ReadGraphPpConfigFromJsonFile(const std::string &file_path, GraphPpConfig &graph_pp_config);
  static Status ReadModelPpConfigFromJsonFile(const std::string &file_path, ModelPpConfig &model_pp_config);
  static Status ReadToolchainFromJsonFile(const std::string &file_path,
                                          std::map<std::string, std::string> &toolchain_map);
  static Status ReadDeployInfoFromJsonFile(const std::string &file_path,
                                           DeployConfigInfo &deploy_conf);
  static Status GetResourceTypeFromNumaConfig(std::set<std::string> &resource_types);

 private:
  static Status ReadCompileConfigJsonFile(const std::string &file_path, nlohmann::json &json_buff);
  static Status CheckFunctionPpConfigJson(const nlohmann::json &json_buff);
  static Status CheckFunctionPpConfigJsonOptional(const nlohmann::json &json_buff);
  static Status CheckFunctionPpConfigFuncListJson(const nlohmann::json &json_buff);
  static Status CheckFunctionPpRunningResourcesInfo(const nlohmann::json &json_buff);
  static Status CheckFunctionPpBufCfg(const nlohmann::json &json_buff);
  static Status CheckFunctionPpCompiler(const nlohmann::json &json_buff);
};

void from_json(const nlohmann::json &json_buff, CompileConfigJson::FunctionPpConfig &func_pp_config);
void from_json(const nlohmann::json &json_buff, CompileConfigJson::FunctionDesc &func_desc);
void from_json(const nlohmann::json &json_buff, CompileConfigJson::GraphPpConfig &graph_pp_config);
void from_json(const nlohmann::json &json_buff, CompileConfigJson::ModelPpConfig &model_pp_config);
void from_json(const nlohmann::json &json_buff, CompileConfigJson::FlowNodeDeployInfo &deploy_info);
void from_json(const nlohmann::json &json_buff, CompileConfigJson::InvokeDeployInfo &invoke_deploy_info);
void from_json(const nlohmann::json &json_buff, CompileConfigJson::FlowNodeBatchDeployInfo &batch_deploy_info);
void from_json(const nlohmann::json &json_buff, CompileConfigJson::TensorDescConfig &tensor_desc_config);
void from_json(const nlohmann::json &json_buff, CompileConfigJson::FlowNodeBatchMemCfg &flow_node_batch_mem_cfg);
void from_json(const nlohmann::json &json_buff, CompileConfigJson::RunningResourceInfo &running_resource_info);
}  // namespace ge
#endif  // AIR_COMPILER_PNE_DATA_FLOW_GRAPH_COMPILE_CONFIG_JSON_H
