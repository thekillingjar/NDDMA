/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_graph_manager.h"
#include <numeric>
#include <regex>
#include <fstream>
#include "mmpa/mmpa_api.h"
#include "common/llm_thread_pool.h"
#include "common/llm_utils.h"
#include "utils/extern_math_util.h"
#include "common/llm_ge_api.h"
#include "common/llm_string_util.h"
#include "common/llm_checker.h"
#include "common/llm_scope_guard.h"

namespace llm {
namespace {
// constexpr const char kLlmFlowGraphDumpName[] = "LlmGraphBuild_";
constexpr const char kFlowGraphNamePrefix[] = "LlmGraph_";
constexpr const char kDataFlowDeployFileSuffix[] = "_dataflow_deploy_info.json";
constexpr const char kPrefixPath[] = "/tmp/";
constexpr const char kLogicDeviceId[] = "logic_device_id";
constexpr const char kLogicDevList[] = "logic_device_list";
constexpr const char kBatchDeployInfo[] = "batch_deploy_info";
constexpr const char kFlowNodeList[] = "flow_node_list";
constexpr const char kListenIpInfo[] = "listen_ip_info";
constexpr const char kIp[] = "ip";
constexpr const char kPort[] = "port";
constexpr const char kCompileConfigFuncList[] = "func_list";
constexpr const char kCompileConfigInputNum[] = "input_num";
constexpr const char kCompileConfigOutputNum[] = "output_num";
constexpr const char kCompileConfigBuiltInFlowFunc[] = "built_in_flow_func";
constexpr const char kCompileConfigFuncName[] = "func_name";
constexpr const char kCompileConfigInputsIndex[] = "inputs_index";
constexpr const char kCompileConfigOutputsIndex[] = "outputs_index";
constexpr const char kFlowNodeNameSuffix[] = "_pp";
constexpr const char kInvokeModelFusionInputs[] = "invoke_model_fusion_inputs";
constexpr uint32_t kDefaultGraphId = 0U;
constexpr uint32_t kMaxBlkSize = 2 * 1024 * 1024;
constexpr const char kNormalPageType[] = "normal";
constexpr const char kHugePageType[] = "huge";
constexpr const char kFuncPpBufCfg[] = "buf_cfg";
constexpr const char kFuncPpBufCfgTotalSize[] = "total_size";
constexpr const char kFuncPpBufCfgBlkSize[] = "blk_size";
constexpr const char kFuncPpBufCfgMaxBufSize[] = "max_buf_size";
constexpr const char kFuncPpBufCfgPageType[] = "page_type";

ge::Status GetDataFlowDeployInfoPath(const std::map<std::string, std::vector<std::string>> &logic_dev_id_to_flow_nodes,
                                     const std::string &file_name, std::string &output_path) {
  nlohmann::ordered_json batch_deploy_info;
  batch_deploy_info[kBatchDeployInfo] = {};
  //  Now that the cluster ID has been identified, iterate through all the logical device ids.
  for (const auto &iter : logic_dev_id_to_flow_nodes) {
    const auto &logic_dev_id = iter.first;
    const auto &flow_node_names = iter.second;
    nlohmann::json json_flow_graph;
    json_flow_graph[kFlowNodeList] = flow_node_names;
    json_flow_graph[kLogicDevList] = logic_dev_id;
    // Add the object to the batch_deploy_info array.
    batch_deploy_info[kBatchDeployInfo].emplace_back(json_flow_graph);
  }
  // Serialize the JSON data to a file.
  output_path = kPrefixPath + file_name;
  std::ofstream output_file(output_path);
  if (!output_file.is_open()) {
    LLMLOGE(ge::FAILED, "can not open %s to write json data.", output_path.c_str());
    return ge::FAILED;
  }
  try {
    output_file << batch_deploy_info;
  } catch (const std::exception &e) {
    LLMLOGE(ge::FAILED, "write json data to %s failed.", output_path.c_str());
    return ge::FAILED;
  }
  LLMLOGI("[GetDataFlowDeployInfoPath] write json data to file:%s", output_path.c_str());
  return ge::SUCCESS;
}
}  // namespace

ge::Status DeployInfo::ParserDeployInfo(const std::map<ge::AscendString, ge::AscendString> &options) {
  auto iter = options.find(LLM_OPTION_CLUSTER_INFO);
  LLM_CHK_BOOL_RET_STATUS(iter != options.cend(), ge::LLM_PARAM_INVALID,
                         "get LLM_OPTION_CLUSTER_INFO failed, options must contain key:%s.", LLM_OPTION_CLUSTER_INFO);
  const nlohmann::ordered_json json = nlohmann::ordered_json::parse(iter->second.GetString());
  LLM_CHK_BOOL_RET_STATUS(json.find(kLogicDeviceId) != json.cend() && json[kLogicDeviceId].is_array(),
                         ge::LLM_PARAM_INVALID, "%s must be array.", kLogicDeviceId);
  for (const auto &logical_device_id : json[kLogicDeviceId]) {
    LLM_CHK_BOOL_RET_STATUS(logical_device_id.is_string(), ge::LLM_PARAM_INVALID, "logical_device_id must be string");
    logical_device_ids_.emplace_back(logical_device_id);
  }

  iter = options.find(LLM_OPTION_ROLE);
  LLM_ASSERT_TRUE(iter != options.end(), "[DeployInfo] not find %s.", LLM_OPTION_ROLE);
  const std::string &role = iter->second.GetString();
  if (json.find(kListenIpInfo) != json.cend()) {
    LLM_CHK_BOOL_RET_STATUS(json[kListenIpInfo].is_array(), ge::LLM_PARAM_INVALID, "%s must be array.", kListenIpInfo);
    for (const auto &ip_info : json[kListenIpInfo]) {
      LLM_CHK_BOOL_RET_STATUS(ip_info.is_object() && (ip_info.find(kIp) != ip_info.cend()) &&
                                 (ip_info.find(kPort) != ip_info.cend()) && ip_info[kIp].is_number_integer() &&
                                 (ip_info[kPort].is_number_integer()),
                             ge::LLM_PARAM_INVALID, "%s is invalid.", kListenIpInfo);
      listen_ips_info_.emplace_back(std::make_pair(ip_info[kIp], ip_info[kPort]));
    }
    LLM_CHK_BOOL_RET_STATUS(logical_device_ids_.size() == listen_ips_info_.size(), ge::LLM_PARAM_INVALID,
                           "logical_device_ids num:%zu not match listen_ips_info num:%zu", logical_device_ids_.size(),
                           listen_ips_info_.size());
  } else {
    LLM_CHK_BOOL_RET_STATUS(role != kPrompt, ge::LLM_PARAM_INVALID, "prompt engine must set listen ip info.");
  }

  LLM_CHK_STATUS_RET(ParseDeviceToRankId(), "Failed to parse rank id");
  LLM_CHK_STATUS_RET(ParseDeviceIndices(options), "Failed to parse device indices");
  return ge::SUCCESS;
}

ge::Status DeployInfo::ParseDeviceToRankId() {
  for (size_t i = 0U; i < logical_device_ids_.size(); ++i) {
    const auto &logical_device_id = logical_device_ids_[i];
    logical_device_id_to_flow_nodes_[logical_device_ids_[i]] = {};
    LLMLOGI("logical device id = %s, rank id = %u.", logical_device_id.c_str());
  }

  return ge::SUCCESS;
}

ge::Status DeployInfo::ParseDeviceIndices(const std::map<ge::AscendString, ge::AscendString> &options) {
  const auto it = options.find(ge::OPTION_EXEC_RANK_ID);
  std::string rank_id_str;
  if (cache_engine_mode_ && (it != options.cend()) && (options.find(ge::OPTION_EXEC_DEVICE_ID) == options.cend())) {
    size_t rank_id = 0U;
    rank_id_str = it->second.GetString();
    LLM_CHK_STATUS_RET(LLMUtils::ToNumber(rank_id_str, rank_id),
                      "Failed to parse rank id: %s", rank_id_str.c_str());
    LLM_CHK_BOOL_RET_STATUS(rank_id < GetDeployedDeviceNum(), ge::LLM_PARAM_INVALID,
                           "rank_id (%zu) out of range [0, %zu)", rank_id, GetDeployedDeviceNum());
    LLMLOGI("rank_id = %zu, will run in SPMD mode, logical_device_id = %s",
           rank_id, GetLogicalDeviceIds()[rank_id].c_str());
    device_indices_.emplace_back(rank_id);
  } else {
    for (size_t device_index = 0U; device_index < GetDeployedDeviceNum(); ++device_index) {
      device_indices_.emplace_back(device_index);
    }
  }
  LLMLOGI("cache_engine_mode = %d, %s = [%s], device_indices = %s",
         static_cast<int32_t>(cache_engine_mode_),
         ge::OPTION_EXEC_RANK_ID,
         rank_id_str.c_str(),
         llm::ToString(device_indices_).c_str());
  return ge::SUCCESS;
}

void DeployInfo::SetCacheEngineMode(bool cache_engine_mode) {
  cache_engine_mode_ = cache_engine_mode;
}

ge::Status ProcessPointCompileConfig::CreateFuncPpCompileConfig(const std::string &compile_config_file) {
  nlohmann::ordered_json invoke_document;
  invoke_document[kCompileConfigFuncList] = nlohmann::ordered_json::array();
  invoke_document[kCompileConfigInputNum] = input_num_;
  invoke_document[kCompileConfigOutputNum] = output_num_;
  invoke_document[kCompileConfigBuiltInFlowFunc] = true;

  uint32_t func_in_num = 0U;
  uint32_t func_out_num = 0U;
  for (const auto &func : udf_func_info_) {
    func_in_num += func.input_indices.size();
    func_out_num += func.output_indices.size();
  }
  if ((func_in_num != input_num_) || (func_out_num != output_num_)) {
    LLMLOGE(ge::LLM_PARAM_INVALID,
           "in/out num between functions and invoke_json setting is different! fun_in_num:%u, fun_out_num:%u, "
           "input_num:%u, output_num:%u", func_in_num, func_out_num, input_num_, output_num_);
    return ge::LLM_PARAM_INVALID;
  }
  // Iterate through each function object, convert it to a JSON object, and add it to the func_list.
  for (const auto &function : udf_func_info_) {
    nlohmann::ordered_json func_json;
    func_json[kCompileConfigFuncName] = function.func_name;

    // Add output_index only when it is not empty.
    if (!function.input_indices.empty()) {
      func_json[kCompileConfigInputsIndex] = function.input_indices;
    }
    if (!function.output_indices.empty()) {
      func_json[kCompileConfigOutputsIndex] = function.output_indices;
    }
    invoke_document[kCompileConfigFuncList].push_back(func_json);
  }
  if (!buf_pool_cfg_.empty()) {
    invoke_document[kFuncPpBufCfg] = nlohmann::ordered_json::array();
    for (const auto &cfg : buf_pool_cfg_) {
      nlohmann::ordered_json cfg_json;
      cfg_json[kFuncPpBufCfgTotalSize] = cfg.total_size;
      cfg_json[kFuncPpBufCfgBlkSize] = cfg.blk_size;
      cfg_json[kFuncPpBufCfgMaxBufSize] = cfg.max_buf_size;
      cfg_json[kFuncPpBufCfgPageType] = cfg.page_type;
      invoke_document[kFuncPpBufCfg].push_back(cfg_json);
    }
  }
  std::ofstream file(compile_config_file);
  if (file.is_open()) {
    file << invoke_document;
    file.close();
    LLMLOGI("json data already write to %s success.", compile_config_file.c_str());
  } else {
    LLMLOGE(ge::LLM_PARAM_INVALID, "can not open file:%s to write json data.", compile_config_file.c_str());
    return ge::LLM_PARAM_INVALID;
  }
  return ge::SUCCESS;
}

ge::Status ProcessPointCompileConfig::CheckBufCfgValue(const BufCfg &buf_cfg) const {
  LLM_ASSERT_TRUE(buf_cfg.total_size != 0U, "Total size can not be zero or larger than UINT32_MAX");
  LLM_ASSERT_TRUE(buf_cfg.max_buf_size != 0U, "max buf size can not be zero or larger than UINT32_MAX");
  LLM_ASSERT_TRUE(buf_cfg.blk_size != 0U, "blk size not be zero or larger than UINT32_MAX");
  LLM_ASSERT_TRUE((buf_cfg.total_size > buf_cfg.max_buf_size) && (buf_cfg.max_buf_size >= buf_cfg.blk_size),
      "The following three params not meet the requirement: total_size[%u] > max_buf_size[%u] >= blk_size[%u]",
      buf_cfg.total_size, buf_cfg.max_buf_size, buf_cfg.blk_size);
    LLM_ASSERT_TRUE(buf_cfg.blk_size != 0, "The blk_size[%u] should not be 0.", buf_cfg.blk_size);
    LLM_ASSERT_TRUE((buf_cfg.blk_size & (buf_cfg.blk_size - 1U)) == 0U, "The blk_size[%u] should be 2^n.",
                   buf_cfg.blk_size);
    LLM_ASSERT_TRUE(buf_cfg.blk_size <= kMaxBlkSize, "The blk_size[%u] should not greater than 2M.", buf_cfg.blk_size);
    LLM_ASSERT_TRUE(buf_cfg.total_size % buf_cfg.blk_size == 0UL,
        "The buffer size[%u] should be multiple of blk_size[%u].", buf_cfg.total_size, buf_cfg.blk_size);
    return ge::SUCCESS;
}

ge::Status ProcessPointCompileConfig::CheckBufPoolCfgJson(nlohmann::json buf_cfg_json) const {
  LLM_ASSERT_TRUE(buf_cfg_json.contains(kFuncPpBufCfgTotalSize), "Config option must contain %s.",
                 kFuncPpBufCfgTotalSize);
  LLM_ASSERT_TRUE(buf_cfg_json[kFuncPpBufCfgTotalSize].is_number_unsigned(),
                 "From config option string %s should be unsigned int type.", kFuncPpBufCfgTotalSize);
  LLM_ASSERT_TRUE(buf_cfg_json.contains(kFuncPpBufCfgBlkSize), "Cfg option must contain %s.", kFuncPpBufCfgBlkSize);
  LLM_ASSERT_TRUE(buf_cfg_json[kFuncPpBufCfgBlkSize].is_number_unsigned(),
                 "From config option string %s should be unsigned int type.", kFuncPpBufCfgBlkSize);
  LLM_ASSERT_TRUE(buf_cfg_json.contains(kFuncPpBufCfgMaxBufSize), "Cfg option must contain %s.",
                 kFuncPpBufCfgMaxBufSize);
  LLM_ASSERT_TRUE(buf_cfg_json[kFuncPpBufCfgMaxBufSize].is_number_unsigned(),
                 "From config option string %s should be unsigned int type.", kFuncPpBufCfgMaxBufSize);
  return ge::SUCCESS;
}

ge::Status ProcessPointCompileConfig::CheckAndConstructBufCfg(
      const std::map<ge::AscendString, ge::AscendString> &options) {
  const auto iter = options.find(LLM_OPTION_BUF_POOL_CFG);
  if (iter == options.end()) {
    LLMLOGI("Buffer pool config is not set.");
    return ge::SUCCESS;
  }
  std::string buf_cfg_str = iter->second.GetString();
  LLM_ASSERT_TRUE(!buf_cfg_str.empty(), "Buffer pool config can not be empty while this option is set.");
  try {
    nlohmann::json cfg_json = nlohmann::json::parse(buf_cfg_str);
    LLM_ASSERT_TRUE(cfg_json.contains(kFuncPpBufCfg),
                   "Buffer pool config must contains buf_cfg list node.");
    const auto &json_buf_cfg = cfg_json[kFuncPpBufCfg];
    LLM_ASSERT_TRUE(json_buf_cfg.is_array(), "buf_cfg should be array.");
    LLM_ASSERT_TRUE(json_buf_cfg.size() > 0, "buf_cfg should contain at least one element.");
    uint32_t last_max_buf_size = 0U;
    for (const auto &cfg_item : json_buf_cfg) {
      LLM_CHK_STATUS_RET(CheckBufPoolCfgJson(cfg_item), "Check buf cfg option json is invalid.");
      BufCfg buf_cfg = {cfg_item[kFuncPpBufCfgTotalSize], cfg_item[kFuncPpBufCfgBlkSize],
                        cfg_item[kFuncPpBufCfgMaxBufSize], kNormalPageType};
      LLM_CHK_STATUS_RET(CheckBufCfgValue(buf_cfg), "Check buffer config value failed.");
      LLM_ASSERT_TRUE(buf_cfg.max_buf_size > last_max_buf_size, "Normal page type should be sorted. "
          "But current size[%zu] is less or equal to last gear[%zu].", buf_cfg.max_buf_size, last_max_buf_size);
      last_max_buf_size = buf_cfg.max_buf_size;
      buf_pool_cfg_.emplace_back(buf_cfg);
      buf_cfg.page_type = kHugePageType;
      buf_pool_cfg_.emplace_back(buf_cfg);
      LLMLOGI("Parse buf pool config success. total size[%u] block size[%u] max buf size[%u]",
             buf_cfg.total_size, buf_cfg.blk_size, buf_cfg.max_buf_size);
    }
  } catch (const nlohmann::json::exception &e) {
    LLMLOGE(ge::LLM_PARAM_INVALID, "Parse buf pool config, exception = %s", e.what());
    return ge::LLM_PARAM_INVALID;
  }
  return ge::SUCCESS;
}

void ProcessPointCompileConfig::AddUdfFuncInfo(const std::string &func_name, const std::vector<uint32_t> &input_indices,
                                               const std::vector<uint32_t> &output_indices) {
  const auto &iter =
      std::find_if(udf_func_info_.cbegin(), udf_func_info_.cend(),
                   [&func_name](const UdfFuncInfo &udf_func_info) { return udf_func_info.func_name == func_name; });
  if (iter == udf_func_info_.cend()) {
    udf_func_info_.emplace_back(UdfFuncInfo{func_name, input_indices, output_indices});
    return;
  }
  LLMLOGI("Udf func:%s already added", func_name.c_str());
}

void FlowGraphManager::CreateFlowDataInfo(const std::string &prefix, const std::vector<uint32_t> &indices,
                                          std::vector<FlowDataInfo> &flow_data_infos) const {
  for (size_t i = 0U; i < indices.size(); ++i) {
    const uint32_t input_index = indices[i];
    LLMLOGD("Begin to create model execute input flow data, prefix = %s, index:%zu", prefix.c_str(), input_index);
    std::string flow_data_name = prefix + std::to_string(input_index);
    ge::TensorDesc input_tensor_desc;
    flow_data_infos.emplace_back(FlowDataInfo{flow_data_name, input_index, input_tensor_desc});
  }
}

ge::dflow::FlowData FlowGraphManager::CreateFlowData(std::string data_name, const int64_t index,
                                                     const ge::TensorDesc &tensor_desc) const {
  auto flow_data = ge::dflow::FlowData(data_name.c_str(), index);
  flow_data.UpdateInputDesc("x", tensor_desc);
  flow_data.UpdateOutputDesc("y", tensor_desc);
  return flow_data;
}

ge::dflow::FlowNode FlowGraphManager::CreateFlowNode(const std::string &node_name, uint32_t input_num,
                                                     uint32_t output_num, const std::string &compile_config,
                                                     const FunctionPpSetter &pp_setter) const {
  std::string pp_name = node_name + kFlowNodeNameSuffix;
  auto pp = ge::dflow::FunctionPp(pp_name.c_str()).SetCompileConfig(compile_config.c_str());
  pp = pp_setter(pp);
  auto node = ge::dflow::FlowNode(node_name.c_str(), input_num, output_num);
  node.AddPp(pp);
  return node;
}
ge::Status FlowGraphManager::BuildFlowGraph(const std::map<ge::AscendString, ge::AscendString> &options,
                                            GeApi *ge_api) {
  std::string dataflow_deploy_info_path;
  std::string
      deploy_node_file_name = std::string(flow_graph_->GetName()) + unique_path_str_ + kDataFlowDeployFileSuffix;
  LLM_CHK_STATUS_RET(GetDataFlowDeployInfoPath(deploy_info_.GetLogicalDeviceIdToFlowNodes(), deploy_node_file_name,
                                              dataflow_deploy_info_path),
                    "[GetDataFlowDeployInfoPath] failed, flow_graph: %s.", flow_graph_->GetName());

  std::map<ge::AscendString, ge::AscendString> graph_options = {
      {OPTION_DATA_FLOW_DEPLOY_INFO_PATH, dataflow_deploy_info_path.c_str()}};
  const auto option_iter = options.find(LLM_OPTION_OUTPUT_MAX_SIZE);
  if (option_iter != options.end()) {
    graph_options[ge::OUTPUT_MAX_SIZE.c_str()] = option_iter->second;
  }

  auto &ge_api_instance = ge_api != nullptr ? *ge_api : GeApi::GetInstance();
  LLM_CHK_STATUS_RET(ge_api_instance.AddGraph(kDefaultGraphId, flow_graph_->ToGeGraph(), graph_options),
                    "[AddForFlowGraph][AddGraph] failed, flow_graph: %s.", flow_graph_->GetName());

  std::vector<ge::Tensor> inputs;
  LLM_CHK_STATUS_RET(ge_api_instance.BuildGraph(kDefaultGraphId, inputs),
                    "[PromptManager] BuildForFlowGraph failed.");
  return ge::SUCCESS;
}

ge::Status FlowGraphManager::ConstructFlowGraph(const FlowNodeDef &flow_node_def,
                                                const std::map<ge::AscendString, ge::AscendString> &options,
                                                const std::vector<int32_t> &device_ids) {
  deploy_info_.SetCacheEngineMode(cache_engine_mode_);
  LLM_CHK_STATUS_RET(deploy_info_.ParserDeployInfo(options));
  std::map<std::string, ge::dflow::FlowOperator> flow_nodes;
  std::vector<ge::dflow::FlowOperator> outputs_operator;
  LLM_CHK_STATUS_RET(GenerateUniquePathStr(flow_node_def, device_ids));
  const std::string &compile_config_file = std::string("/tmp/invoke_func") + unique_path_str_ + ".json";
  ProcessPointCompileConfig process_point_compile_config = ProcessPointCompileConfig(
      flow_node_def.NumInputs(), flow_node_def.NumOutputs());
  for (const auto &flow_func_def : flow_node_def.flow_func_defs) {
    process_point_compile_config.AddUdfFuncInfo(flow_func_def.func_name,
                                                flow_func_def.input_indices,
                                                flow_func_def.output_indices);
  }
  LLM_CHK_STATUS_RET(process_point_compile_config.CheckAndConstructBufCfg(options),
      "Construct buffer pool config by option failed.");
  const auto ret = process_point_compile_config.CreateFuncPpCompileConfig(compile_config_file);
  LLM_CHK_BOOL_RET_STATUS(ret == ge::SUCCESS, ret, "create func pp compile config failed");

  std::vector<FlowDataInfo> flow_data_infos;
  for (const auto &flow_func_def : flow_node_def.flow_func_defs) {
    CreateFlowDataInfo(flow_func_def.input_name_prefix, flow_func_def.input_indices, flow_data_infos);
  }
  std::vector<ge::dflow::FlowOperator> inputs_operator;
  for (const auto &flow_data_info : flow_data_infos) {
    auto flow_data = CreateFlowData(flow_data_info.name, flow_data_info.index, flow_data_info.tensor_desc);
    flow_nodes.emplace(flow_data_info.name, flow_data);
    inputs_operator.emplace_back(flow_data);
  }

  // 创建flow node节点
  auto flow_node = CreateFlowNode(flow_node_def.name, flow_node_def.NumInputs(), flow_node_def.NumOutputs(),
                                  compile_config_file, flow_node_def.pp_setter);
  LLM_ASSERT_TRUE(flow_nodes.emplace(flow_node_def.name, flow_node).second);
  outputs_operator.emplace_back(flow_node);
  const std::string deployed_logical_device =
      deploy_info_.GetLogicalDeviceIds()[flow_node_def.logical_device_index];
  LLMLOGI("Add node:%s to device_index=%zu, logical_device_id=%s",
         flow_node_def.name.c_str(), flow_node_def.logical_device_index, deployed_logical_device.c_str());
  deploy_info_.GetLogicalDeviceIdToFlowNodes()[deployed_logical_device].emplace_back(flow_node_def.name);
  // 添加输入
  for (size_t i = 0U; i < flow_node_def.NumInputs(); ++i) {
    flow_node.SetInput(i, inputs_operator[i], 0U);
  }
  auto graph_name = kFlowGraphNamePrefix + std::to_string(cluster_id_);
  if (deploy_info_.GetDeviceIndices().size() < deploy_info_.GetDeployedDeviceNum()) {
    graph_name += ("_" + std::to_string(deploy_info_.GetDeviceIndices().front()));
  }
  flow_graph_ = llm::MakeShared<ge::dflow::FlowGraph>(graph_name.c_str());
  LLM_CHECK_NOTNULL(flow_graph_);
  flow_graph_->SetInputs(inputs_operator).SetOutputs(outputs_operator);
  return ge::SUCCESS;
}

void FlowGraphManager::SetCacheEngineMode(bool cache_engine_mode) {
  cache_engine_mode_ = cache_engine_mode;
}

ge::Status FlowGraphManager::GenerateUniquePathStr(const FlowNodeDef &flow_node_def,
                                                   const std::vector<int32_t> &device_ids) {
  std::stringstream ss;
  ss << (flow_node_def.is_prompt ? "_prompt_" : "_decoder_") << std::to_string(cluster_id_) << "_";
  if (device_ids.empty()) {
    ss << flow_node_def.logical_device_index;
  } else {
    for (size_t i = 0U; i < device_ids.size(); ++i) {
      ss << device_ids[i];
      if (i != device_ids.size() - 1U) {
        ss << "_";
      }
    }
  }
  unique_path_str_ = ss.str();
  return ge::SUCCESS;
}
}  // namespace llm
