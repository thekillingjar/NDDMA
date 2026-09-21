/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_LLM_ENGINE_LLM_ENGINE_API_H
#define AIR_RUNTIME_LLM_ENGINE_LLM_ENGINE_API_H
#include <unordered_map>
#include "ge/ge_api_types.h"
#include "ge/ge_api.h"
#include "graph/graph.h"
#include "ge/ge_graph_compile_summary.h"

namespace llm {
class GeApi {
 public:
  static GeApi &GetInstance();
  virtual ~GeApi() = default;
  virtual ge::Status Initialize(const std::map<ge::AscendString, ge::AscendString> &options);
  virtual ge::Status Finalize();
  ge::Status SafeFinalize();
  virtual ge::Status AddGraph(uint32_t graph_id, const ge::Graph &graph,
                              const std::map<ge::AscendString, ge::AscendString> &options);
  virtual ge::Status BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs);
  virtual ge::Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indices,
                                       const std::vector<ge::Tensor> &inputs, const ge::DataFlowInfo &info,
                                       int32_t timeout);
  virtual ge::Status FeedRawData(uint32_t graph_id, const uint32_t &index, const std::vector<ge::Tensor> &inputs,
                                 const ge::DataFlowInfo &info, int32_t timeout);
  virtual ge::Status FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indices,
                                        std::vector<ge::Tensor> &outputs, ge::DataFlowInfo &info, int32_t timeout);
  virtual ge::Status RunGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs,
                              std::vector<ge::Tensor> &outputs);
  virtual ge::CompiledGraphSummaryPtr GetCompiledGraphSummary(uint32_t graph_id);
  virtual void DestroySession();
  virtual size_t NumSessions() const;

  virtual std::unique_ptr<GeApi> NewSession(const std::map<ge::AscendString, ge::AscendString> &options);
 private:
  static std::unique_ptr<GeApi> instance_;
  std::shared_ptr<ge::Session> sess_ = nullptr;
};
}  // namespace llm

#endif  // AIR_RUNTIME_LLM_ENGINE_LLM_ENGINE_API_H
