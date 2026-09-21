/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_engine_test_helper.h"
#include "llm_string_util.h"
#include "llm_test_helper.h"

namespace ge {
namespace {
std::set<int64_t> g_sessions;
std::mutex g_session_mu_;
}  // namespace

void LlmEngineOptionBuilder::SetKvCacheOptions() {
  const uint32_t kv_size = num_stages_;
  std::string ref_shapes;
  std::string ref_dtype;
  for (uint32_t i = 0U; i < kv_size; i++) {
    if (i < kv_size - 1U) {
      ref_shapes.append("8192,8192;");
      ref_dtype.append("1;");
    } else {
      ref_shapes.append("8192,8192");
      ref_dtype.append("1");
    }
  }
}

void LlmEngineOptionBuilder::SetDeployInfoOptions() {
  std::vector<std::string> logic_device_id_list;
  std::vector<std::string> listen_ip_info_list;
  for (int32_t i = 0; i < num_stages_; ++i) {
    logic_device_id_list.emplace_back(std::string("\"0:") + std::to_string(i) + ":" + "0:0\"");
    logic_device_id_list.emplace_back(std::string("\"0:") + std::to_string(i) + ":" + "1:0\"");
    listen_ip_info_list.emplace_back(R"({"ip":176163329,"port":6000})");
    listen_ip_info_list.emplace_back(R"({"ip":176163330,"port":6000})");
  }
  const auto &logic_device_ids = llm::StringUtils::Join(logic_device_id_list.cbegin(), logic_device_id_list.cend(), ",");
  const auto &listen_ip_infos = llm::StringUtils::Join(listen_ip_info_list.cbegin(), listen_ip_info_list.cend(), ",");
  std::string
      deploy_cluster_path = std::string(R"({"cluster_id":0,"logic_device_id":[)") + logic_device_ids + "]";
  deploy_cluster_path += (",\"listen_ip_info\":[" + listen_ip_infos + "]}");
  llm_options_[llm::LLM_OPTION_CLUSTER_INFO] = deploy_cluster_path.c_str();
}

std::map<ge::AscendString, ge::AscendString> &LlmEngineOptionBuilder::Build() {
  std::vector<std::string> om_cache_path_list(num_stages_ * 2, "./");
  const auto &om_cache_paths = llm::StringUtils::Join(om_cache_path_list.cbegin(), om_cache_path_list.cend(), ";");
  std::string hcom_cluster_config =
      R"({"comm_group":[{"group_name": "tp_group_0", "group_rank_list" : "[0, 1]"}, {"group_name": "tp_group_1", "group_rank_list" : "[2, 3]"}], "rank_table": "rank_table"})";
  char_t numa_config_path[4096];
  (void) realpath("../tests/dflow/llm_datadist/st/testcase/llm_datadist/json_file/numa_config_decoder.json", numa_config_path);
  // 构建选项参数
  llm_options_ = {
      {"ge.socVersion", "Ascend910B1"},
      {"ge.graphRunMode", "0"},
      {llm::LLM_OPTION_ROLE, role_.c_str()},
      {"ge.distributed_cluster_build", "1"},
      {"RESOURCE_CONFIG_PATH", numa_config_path},
      {llm::LLM_OPTION_SYNC_KV_CACHE_WAIT_TIME, "20"},
  };

  SetDeployInfoOptions();
  SetKvCacheOptions();

  return llm_options_;
}

LlmEngineOptionBuilder &LlmEngineOptionBuilder::Role(llm::RoleType role_type) {
  role_ = role_type == llm::RoleType::PROMPT ? "Prompt" : "Decoder";
  return *this;
}


LlmEngineOptionBuilder &LlmEngineOptionBuilder::NumStages(int32_t num_stages) {
  num_stages_ = num_stages;
  return *this;
}

ge::Status DefaultGeApiStub::Initialize(const std::map<ge::AscendString, ge::AscendString> &options) {
  LLMLOGI("Stub Initialize");
  g_sessions.clear();
  g_sessions.emplace(session_id_);
  return ge::SUCCESS;
}

ge::Status DefaultGeApiStub::Finalize() {
  LLMLOGI("Stub Finalize");
  g_sessions.erase(session_id_);
  return ge::SUCCESS;
}

ge::Status DefaultGeApiStub::AddGraph(uint32_t graph_id,
                                      const Graph &graph,
                                      const std::map<ge::AscendString, ge::AscendString> &options) {
  LLMLOGI("Stub AddGraph");
  return ge::SUCCESS;
}

ge::Status DefaultGeApiStub::BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs) {
  LLMLOGI("Stub BuildGraph");
  return ge::SUCCESS;
}

ge::Status DefaultGeApiStub::FeedDataFlowGraph(uint32_t graph_id,
                                               const std::vector<uint32_t> &indices,
                                               const std::vector<ge::Tensor> &inputs,
                                               const DataFlowInfo &info,
                                               int32_t timeout) {
  LLMEVENT("Stub FeedDataFlowGraph, indices = %s", llm::ToString(indices).c_str());
  if (inputs.size() != indices.size()) {
    LLMLOGE(FAILED, "num_inputs (%zu) mismatches that of indices (%zu)", inputs.size(), indices.size());
    return ge::FAILED;
  }
  if (indices[0] == UINT32_MAX) {
    return ge::FAILED;
  }
  LLMLOGI("Stub FeedDataFlowGraph");
  return ge::SUCCESS;
}

ge::Status DefaultGeApiStub::FeedRawData(uint32_t graph_id, const uint32_t &index, const std::vector<ge::Tensor> &inputs,
                                         const ge::DataFlowInfo &info, int32_t timeout) {
  LLMLOGI("Stub FeedRawData");
  return ge::SUCCESS;
}

ge::Status DefaultGeApiStub::FetchDataFlowGraph(uint32_t graph_id,
                                                const std::vector<uint32_t> &indexes,
                                                std::vector<ge::Tensor> &outputs,
                                                DataFlowInfo &info,
                                                int32_t timeout) {
  if (indexes.empty()) {
    return ge::SUCCESS;
  }
  if (indexes[0] == UINT32_MAX) {
    return ge::FAILED;
  }
  for (size_t i = 0; i < indexes.size(); i++) {
    ge::Tensor output_tensor;
    llm::LLMTestUtils::BuildInput(output_tensor, {2, 8192}, 2 * 8192);
    outputs.push_back(output_tensor);
  }
  LLMLOGI("Stub FetchDataFlowGraph");
  return ge::SUCCESS;
}

ge::Status DefaultGeApiStub::RunGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs) {
  LLMLOGI("Stub RunGraph");
  ge::Tensor output_tensor;
  llm::LLMTestUtils::BuildInput(output_tensor, {8192, 8192}, 8192 * 8192);
  outputs.push_back(output_tensor);
  return ge::SUCCESS;
}

ge::CompiledGraphSummaryPtr DefaultGeApiStub::GetCompiledGraphSummary(uint32_t graph_id) {
  LLMLOGI("Stub RunGraph");
  ge::CompiledGraphSummaryPtr summary = nullptr;
  return summary;
}

ge::Status DefaultGeApiStub::LoadGraph(const uint32_t graph_id,
                                       const std::map<ge::AscendString, ge::AscendString> &options,
                                       const string &om_file_path) {
  return ge::SUCCESS;
}

std::unique_ptr<llm::GeApi> DefaultGeApiStub::NewSession(const std::map<ge::AscendString, ge::AscendString> &options) {
  auto session_id = ++session_id_gen_;
  return llm::MakeUnique<DefaultGeApiStub>(session_id);
}

DefaultGeApiStub::DefaultGeApiStub(int64_t session_id) : session_id_(session_id) {
  std::lock_guard<std::mutex> lk(g_session_mu_);
  g_sessions.emplace(session_id);
}

DefaultGeApiStub::~DefaultGeApiStub() {
//  std::lock_guard<std::mutex> lk(g_session_mu_);
//  g_sessions.erase(session_id_);
}

void DefaultGeApiStub::DestroySession() {
  std::lock_guard<std::mutex> lk(g_session_mu_);
  g_sessions.erase(session_id_);
  GeApi::DestroySession();
}

size_t DefaultGeApiStub::NumSessions() const {
  std::lock_guard<std::mutex> lk(g_session_mu_);
  return g_sessions.size();
}
}  // namespace ge
