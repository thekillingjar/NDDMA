/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_ge_api.h"
#include "ge/ge_data_flow_api.h"
#include "common/llm_utils.h"
#include "common/llm_checker.h"
#include "session/session_utils.h"
#include "common/llm_scope_guard.h"

namespace llm {
namespace {
const char *const RESOURCE_CONFIG_PATH = "ge.resourceConfigPath";
}

std::unique_ptr<GeApi> GeApi::instance_;
ge::Status GeApi::Initialize(const std::map<ge::AscendString, ge::AscendString> &options) {
  auto init_options = options;
  init_options.erase(RESOURCE_CONFIG_PATH);
  LLM_CHK_STATUS_RET(ge::GEInitialize(init_options));
  sess_ = llm::MakeShared<ge::Session>(options);
  LLM_CHECK_NOTNULL(sess_);
  return ge::SUCCESS;
}

ge::Status GeApi::Finalize() {
  DestroySession();
  return ge::GEFinalize();
}

ge::Status GeApi::SafeFinalize() {
  if (NumSessions() == 0U) {
    LLMLOGI("Already finalized by others");
    return ge::SUCCESS;
  }
  DestroySession();
  if (NumSessions() == 0U) {
    LLMLOGI("session number = 0, start to finalize");
    return Finalize();
  }
  // 如果LLM申请的session都释放后还有session，说明存在另一个前端还在使用，留给它们去Finalize GE
  LLMLOGI("Still referenced by others, do not invoke GEFinalize");
  return ge::SUCCESS;
}

void GeApi::DestroySession() {
  sess_.reset();
}

ge::Status GeApi::AddGraph(uint32_t graph_id, const ge::Graph &graph,
                           const std::map<ge::AscendString, ge::AscendString> &options) {
  return sess_->AddGraph(graph_id, graph, options);
}

GeApi &GeApi::GetInstance() {
  if (instance_ != nullptr) {
    return *instance_;
  }
  static GeApi instance;
  return instance;
}

ge::Status GeApi::BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs) {
  return sess_->BuildGraph(graph_id, inputs);
}

ge::Status GeApi::FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indices,
                                    const std::vector<ge::Tensor> &inputs, const ge::DataFlowInfo &info,
                                    int32_t timeout) {
  return sess_->FeedDataFlowGraph(graph_id, indices, inputs, info, timeout);
}

ge::Status GeApi::FeedRawData(uint32_t graph_id, const uint32_t &index, const std::vector<ge::Tensor> &inputs,
                              const ge::DataFlowInfo &info, int32_t timeout) {
  std::vector<ge::RawData> raw_datas;
  for (const auto &input : inputs) {
    ge::RawData usr_raw_data = {.addr = static_cast<const void *>(input.GetData()), .len = input.GetSize()};
    raw_datas.emplace_back(std::move(usr_raw_data));
  }

  return sess_->FeedRawData(graph_id, raw_datas, index, info, timeout);
}

ge::Status GeApi::FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indices,
                                     std::vector<ge::Tensor> &outputs, ge::DataFlowInfo &info, int32_t timeout) {
  return sess_->FetchDataFlowGraph(graph_id, indices, outputs, info, timeout);
}

ge::Status GeApi::RunGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs) {
  return sess_->RunGraph(graph_id, inputs, outputs);
}

ge::CompiledGraphSummaryPtr GeApi::GetCompiledGraphSummary(uint32_t graph_id) {
  return sess_->GetCompiledGraphSummary(graph_id);
}

std::unique_ptr<GeApi> GeApi::NewSession(const std::map<ge::AscendString, ge::AscendString> &options) {
  auto new_ge_api = llm::MakeUnique<GeApi>();
  LLM_ASSERT_NOTNULL(new_ge_api);
  new_ge_api->sess_ = llm::MakeShared<ge::Session>(options);
  LLM_ASSERT_NOTNULL(new_ge_api->sess_);
  return new_ge_api;
}

size_t GeApi::NumSessions() const {
  return ge::SessionUtils::NumSessions();
}
}  // namespace llm
