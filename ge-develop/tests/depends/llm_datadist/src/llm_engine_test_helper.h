/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TESTS_DEPENDS_LLM_ENGINE_SRC_LLM_ENGINE_LLM_ENGINE_TEST_HELPER_H_
#define AIR_TESTS_DEPENDS_LLM_ENGINE_SRC_LLM_ENGINE_LLM_ENGINE_TEST_HELPER_H_

#include <vector>
#include <cstdlib>
#include "llm_datadist/llm_error_codes.h"
#include "ge/ge_api_types.h"
#include "ge/ge_api.h"
#include "ge/ge_ir_build.h"
#include "common/llm_ge_api.h"
#include "common/llm_utils.h"
#include "common/mem_utils.h"
#include "dlog_pub.h"
#include <nlohmann/json.hpp>
#include "common/llm_log.h"
#include "common/llm_checker.h"
#include "llm_test_helper.h"

namespace ge {
class LlmEngineOptionBuilder {
 public:
  std::map<ge::AscendString, ge::AscendString> &Build();

  LlmEngineOptionBuilder &Role(llm::RoleType role_type);
  LlmEngineOptionBuilder &NumStages(int32_t num_stages);

 private:
  void SetDeployInfoOptions();
  void SetKvCacheOptions();

  int32_t num_stages_ = 1;
  std::string role_;
  std::map<ge::AscendString, ge::AscendString> llm_options_;
};

class DefaultGeApiStub : public llm::GeApi {
 public:
  DefaultGeApiStub(int64_t session_id = 0);
  ~DefaultGeApiStub();
  ge::Status Initialize(const std::map<ge::AscendString, ge::AscendString> &options) override;
  ge::Status Finalize() override;
  ge::Status AddGraph(uint32_t graph_id, const ge::Graph &graph,
                      const std::map<ge::AscendString, ge::AscendString> &options) override;
  ge::Status BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs) override;
  ge::Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indices,
                               const std::vector<ge::Tensor> &inputs, const ge::DataFlowInfo &info,
                               int32_t timeout) override;
  ge::Status FeedRawData(uint32_t graph_id, const uint32_t &index, const std::vector<ge::Tensor> &inputs,
                         const ge::DataFlowInfo &info, int32_t timeout) override;
  ge::Status FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                std::vector<ge::Tensor> &outputs, ge::DataFlowInfo &info, int32_t timeout) override;
  ge::Status RunGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs,
                      std::vector<ge::Tensor> &outputs) override;

  ge::CompiledGraphSummaryPtr GetCompiledGraphSummary(uint32_t graph_id);

  ge::Status LoadGraph(const uint32_t graph_id, const std::map<ge::AscendString, ge::AscendString> &options,
                       const std::string &om_file_path);
  std::unique_ptr<GeApi> NewSession(const std::map<ge::AscendString, ge::AscendString> &options) override;

  size_t NumSessions() const override;

  void DestroySession() override;
 private:
  int64_t session_id_gen_ = 0;
  int64_t session_id_ = 0;
};
}  // namespace ge

#endif  // AIR_TESTS_DEPENDS_LLM_ENGINE_SRC_LLM_ENGINE_LLM_ENGINE_TEST_HELPER_H_
