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
#include "session/session_utils.h"

namespace ge {
namespace {
std::set<int64_t> g_sessions;
std::mutex g_session_mu_;
uint64_t session_id_gen_ = 0;
}  // namespace

ge::Status Session::AddGraph(uint32_t graph_id,
                                      const Graph &graph,
                                      const std::map<ge::AscendString, ge::AscendString> &options) {
  LLMLOGI("Stub AddGraph");
  return ge::SUCCESS;
}

ge::Status Session::BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs) {
  LLMLOGI("Stub BuildGraph");
  return ge::SUCCESS;
}

ge::Status Session::FeedDataFlowGraph(uint32_t graph_id,
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

ge::Status Session::FeedRawData(uint32_t graph_id, const std::vector<RawData> &raw_data_list, const uint32_t index,
                    const DataFlowInfo &info, int32_t timeout) {
  LLMLOGI("Stub FeedRawData");
  return ge::SUCCESS;
}

ge::Status Session::FetchDataFlowGraph(uint32_t graph_id,
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

ge::Status Session::RunGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs) {
  LLMLOGI("Stub RunGraph");
  ge::Tensor output_tensor;
  llm::LLMTestUtils::BuildInput(output_tensor, {8192, 8192}, 8192 * 8192);
  outputs.push_back(output_tensor);
  return ge::SUCCESS;
}

ge::CompiledGraphSummaryPtr Session::GetCompiledGraphSummary(uint32_t graph_id) {
  LLMLOGI("Stub RunGraph");
  ge::CompiledGraphSummaryPtr summary = nullptr;
  return summary;
}

ge::Status Session::LoadGraph(const uint32_t graph_id, const std::map<AscendString, AscendString> &options,
                   void *stream) const {
  LLMLOGI("Stub LoadGraph");
  return ge::SUCCESS;
}

Session::Session(const std::map<std::string, std::string> &options) {
  std::lock_guard<std::mutex> lk(g_session_mu_);
  sessionId_ = ++session_id_gen_;
  g_sessions.emplace(sessionId_);
}

Session::Session(const std::map<AscendString, AscendString> &options) {
  std::lock_guard<std::mutex> lk(g_session_mu_);
  sessionId_ = ++session_id_gen_;
  g_sessions.emplace(sessionId_);
}

Session::~Session() {
    std::lock_guard<std::mutex> lk(g_session_mu_);
    g_sessions.erase(sessionId_);
}

size_t SessionUtils::NumSessions() {
  LLMLOGI("Stub SessionUtils::NumSessions");
  std::lock_guard<std::mutex> lk(g_session_mu_);
  return g_sessions.size();
}
}  // namespace ge
