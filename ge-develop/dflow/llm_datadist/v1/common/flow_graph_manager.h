/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_LLM_ENGINE_COMMON_FLOW_GRAPH_MANAGER_H_
#define AIR_RUNTIME_LLM_ENGINE_COMMON_FLOW_GRAPH_MANAGER_H_

#include <map>
#include "ge/ge_ir_build.h"
#include "ge/ge_api_types.h"
#include "flow_graph/flow_graph.h"
#include "common/llm_ge_api.h"
#include "nlohmann/json.hpp"

namespace llm {
constexpr size_t kDefaultModelNum = 1UL;
constexpr uint32_t kDefaultInputNum = 1U;
constexpr uint32_t kDefaultOutputNum = 1U;
constexpr uint64_t kDefaultBatchSize = 1UL;
constexpr uint64_t kDefaultPagedAttentionBlockSize = 16UL;

using FunctionPpSetter = std::function<ge::dflow::FunctionPp(ge::dflow::FunctionPp)>;

class DeployInfo {
 public:
  ge::Status ParserDeployInfo(const std::map<ge::AscendString, ge::AscendString> &options);

  std::map<std::string, std::vector<std::string>> &GetLogicalDeviceIdToFlowNodes() {
    return logical_device_id_to_flow_nodes_;
  }

  size_t GetDeployedDeviceNum() const {
    return logical_device_ids_.size();
  }
  const std::vector<size_t> &GetDeviceIndices() const {
    return device_indices_;
  }

  const std::vector<std::string> &GetLogicalDeviceIds() const {
    return logical_device_ids_;
  }

  const std::vector<std::pair<int64_t, int64_t>> &GetListenIpsInfo() const {
    return listen_ips_info_;
  }

  void SetCacheEngineMode(bool cache_engine_mode);

 private:
  ge::Status ParseDeviceToRankId();
  ge::Status ParseDeviceIndices(const std::map<ge::AscendString, ge::AscendString> &options);
  std::vector<std::string> logical_device_ids_;
  std::vector<std::pair<int64_t, int64_t>> listen_ips_info_;
  std::vector<size_t> device_indices_;
  std::map<std::string, std::vector<std::string>> logical_device_id_to_flow_nodes_;
  bool cache_engine_mode_ = false;
};

struct UdfFuncInfo {
  std::string func_name;
  std::vector<uint32_t> input_indices;
  std::vector<uint32_t> output_indices;
};

struct FlowDataInfo {
  std::string name;
  uint32_t index;
  ge::TensorDesc tensor_desc;
};

struct BufCfg {
  uint32_t total_size;
  uint32_t blk_size;
  uint32_t max_buf_size;
  std::string page_type;
};

struct FlowNodeInfo {
  std::string name;
  uint32_t input_num;
  uint32_t output_num;
  std::string compile_config;
  FunctionPpSetter pp_setter;
  int32_t deployed_logical_device;
  std::vector<std::string> src_names;
  std::vector<uint32_t> src_indices;
  std::vector<uint32_t> dst_indices;
  bool as_output;
};

class ProcessPointCompileConfig {
 public:
  ProcessPointCompileConfig(const uint32_t input_num, const uint32_t output_num)
      : input_num_(input_num), output_num_(output_num) {}

  ge::Status CreateFuncPpCompileConfig(const std::string &compile_config_file);
  void AddUdfFuncInfo(const std::string &func_name, const std::vector<uint32_t> &input_indices,
                      const std::vector<uint32_t> &output_indices);
  ge::Status CheckAndConstructBufCfg(const std::map<ge::AscendString, ge::AscendString> &options);
 private:
  ge::Status CheckBufCfgValue(const BufCfg &buf_cfg) const;
  ge::Status CheckBufPoolCfgJson(nlohmann::json buf_cfg_json) const;

  uint32_t input_num_;
  uint32_t output_num_;
  std::vector<UdfFuncInfo> udf_func_info_;
  std::vector<BufCfg> buf_pool_cfg_;
};

struct FlowFuncDef {
  std::string func_name;
  std::string input_name_prefix;
  std::vector<uint32_t> input_indices;
  std::vector<uint32_t> output_indices;
};

struct FlowNodeDef {
  bool is_prompt = false;
  std::string role;
  std::string name;
  size_t logical_device_index;
  std::vector<FlowFuncDef> flow_func_defs;
  FunctionPpSetter pp_setter;
  size_t NumInputs() const {
    size_t num_inputs = 0;
    for (const auto &flow_func_def : flow_func_defs) {
      num_inputs += flow_func_def.input_indices.size();
    }
    return num_inputs;
  }

  size_t NumOutputs() const {
    size_t num_outputs = 0;
    for (const auto &flow_func_def : flow_func_defs) {
      num_outputs += flow_func_def.output_indices.size();
    }
    return num_outputs;
  }
};

class FlowGraphManager {
 public:
  explicit FlowGraphManager(const uint64_t cluster_id) : cluster_id_(cluster_id) {}

  ge::Status BuildFlowGraph(const std::map<ge::AscendString, ge::AscendString> &options, GeApi *ge_api = nullptr);

  ge::Status ConstructFlowGraph(const FlowNodeDef &flow_node_def,
                                const std::map<ge::AscendString, ge::AscendString> &options,
                                const std::vector<int32_t> &device_ids);

  size_t GetDeployedDeviceNum() const {
    return deploy_info_.GetDeployedDeviceNum();
  }

  const std::vector<std::pair<int64_t, int64_t>> &GetListenIpsInfo() const {
    return deploy_info_.GetListenIpsInfo();
  }

  void CreateFlowDataInfo(const std::string &prefix, const std::vector<uint32_t> &indices,
                          std::vector<FlowDataInfo> &flow_data_infos) const;

  void SetCacheEngineMode(bool cache_engine_mode);
 private:
  ge::dflow::FlowData CreateFlowData(std::string data_name, const int64_t index,
                                     const ge::TensorDesc &tensor_desc) const;
  ge::dflow::FlowNode CreateFlowNode(const std::string &node_name, uint32_t input_num, uint32_t output_num,
                                     const std::string &compile_config, const FunctionPpSetter &pp_setter) const;
  ge::Status GenerateUniquePathStr(const FlowNodeDef &flow_node_def,
                                   const std::vector<int32_t> &device_ids);

  uint64_t cluster_id_;
  bool cache_engine_mode_ = false;
  DeployInfo deploy_info_;
  std::string unique_path_str_;
  std::vector<std::vector<std::string>> pp_model_paths_;
  std::shared_ptr<ge::dflow::FlowGraph> flow_graph_{nullptr};
};
}  // namespace llm
#endif  // AIR_RUNTIME_LLM_ENGINE_COMMON_FLOW_GRAPH_MANAGER_H_
