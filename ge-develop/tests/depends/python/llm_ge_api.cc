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
#include "llm_utils.h"
#include "framework/common/debug/ge_log.h"
#include "common/mem_utils.h"
#include "common/llm_inner_types.h"
#include "cache_engine_stubs.h"
#include "llm_test_helper.h"
#include "common/llm_log.h"

namespace llm {

ge::Status GeApi::Initialize(const std::map<ge::AscendString, ge::AscendString> &options) {
  LLMLOGI("Stub GE Initialize");
  return ge::SUCCESS;
}

ge::Status GeApi::Finalize() {
  return ge::SUCCESS;
}

ge::Status GeApi::AddGraph(uint32_t graph_id, const ge::Graph &graph,
                           const std::map<ge::AscendString, ge::AscendString> &options) {
  LLMLOGI("Stub AddGraph");
  return ge::SUCCESS;
}

GeApi &GeApi::GetInstance() {
  if (instance_ != nullptr) {
    return *instance_;
  }
  static GeApi instance;
  return instance;
}

ge::Status GeApi::BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs) {
  LLMLOGI("Stub BuildGraph");
  return ge::SUCCESS;
}

ge::Status GeApi::FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indices,
                                    const std::vector<ge::Tensor> &inputs, const ge::DataFlowInfo &info,
                                    int32_t timeout) {
  LLMLOGI("Stub FeedDataFlowGraph");
  return ge::SUCCESS;
}

ge::Status GeApi::FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                     std::vector<ge::Tensor> &outputs, ge::DataFlowInfo &info, int32_t timeout) {
  LLMLOGI("Stub FetchDataFlowGraph");
  for (size_t i = 0; i < indexes.size(); i++) {
    ge::Tensor output_tensor;
    llm::LLMTestUtils::BuildInput(output_tensor, {1, 256}, 1 * 256);
    outputs.push_back(output_tensor);
  }
  return ge::SUCCESS;
}

ge::Status GeApi::RunGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs) {
  LLMLOGI("Stub RunGraph");
  ge::Tensor output_tensor;
  llm::LLMTestUtils::BuildInput(output_tensor, {1, 256}, 1 * 256);
  outputs.push_back(output_tensor);
  return ge::SUCCESS;
}

ge::CompiledGraphSummaryPtr GeApi::GetCompiledGraphSummary(uint32_t graph_id) {
  ge::CompiledGraphSummaryPtr summary = nullptr;
  return summary;
}

std::unique_ptr<GeApi> GeApi::NewSession(const std::map<ge::AscendString, ge::AscendString> &options) {
  dlog_setlevel(0, 0, 0);
  auto new_ge_api = llm::MakeUnique<CacheEngineGeApi>();
  return new_ge_api;
}
}  // namespace llm
