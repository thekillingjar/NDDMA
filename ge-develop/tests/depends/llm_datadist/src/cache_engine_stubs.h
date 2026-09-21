/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TESTS_DEPENDS_LLM_ENGINE_SRC_CACHE_ENGINE_STUBS_H_
#define AIR_TESTS_DEPENDS_LLM_ENGINE_SRC_CACHE_ENGINE_STUBS_H_

#include <vector>

#include "common/cache_engine.h"
#include "llm_datadist.h"
#include "llm_flow_service.h"

namespace llm {
class CacheEngineGeApi : public llm::GeApi {
 public:
  CacheEngineGeApi();
  ~CacheEngineGeApi();

  ge::Status Initialize(const std::map<ge::AscendString, ge::AscendString> &options) override;
  ge::Status Finalize() override;
  ge::Status AddGraph(uint32_t graph_id, const ge::Graph &graph,
                      const std::map<ge::AscendString, ge::AscendString> &options) override;
  ge::Status BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs) override;

  ge::Status FeedDataFlowGraph(uint32_t graph_id,
                               const std::vector<uint32_t> &indices,
                               const std::vector<ge::Tensor> &inputs,
                               const ge::DataFlowInfo &info,
                               int32_t timeout) override;

  ge::Status FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                std::vector<ge::Tensor> &outputs, ge::DataFlowInfo &info, int32_t timeout) override;

  std::unique_ptr<GeApi> NewSession(const std::map<ge::AscendString, ge::AscendString> &options) override {
    return llm::MakeUnique<CacheEngineGeApi>();
  }

  std::map<uint32_t, std::vector<ge::Tensor>> index_to_output_tensors_;
  std::vector<ge::Tensor> cache_get_tensors_;
  class MockFlowNode;
  std::unique_ptr<MockFlowNode> udf_node_;
  std::vector<uint64_t> transaction_ids_ = std::vector<uint64_t>(static_cast<size_t>(FlowFuncType::kMax), 1U);
};
}

#endif  // AIR_TESTS_DEPENDS_LLM_ENGINE_SRC_CACHE_ENGINE_STUBS_H_
