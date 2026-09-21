/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUILT_IN_LLM_FUNC_LLM_SERVICE_FLOW_FUNC_
#define BUILT_IN_LLM_FUNC_LLM_SERVICE_FLOW_FUNC_

#include <mutex>
#include "flow_func/meta_multi_func.h"
#include "flow_func/meta_params.h"
#include "flow_func/meta_run_context.h"
#include "llm_common/cache_manager.h"
#include "llm_common/cluster_manager.h"

namespace FlowFunc {
class LlmServiceFlowFunc : public MetaMultiFunc {
public:
    ~LlmServiceFlowFunc() override;

    int32_t Init(const std::shared_ptr<MetaParams> &params) override;
    int32_t Initialize(const std::shared_ptr<MetaRunContext> &run_context,
                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t UpdateLink(const std::shared_ptr<MetaRunContext> &run_context,
                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t UnlinkData(const std::shared_ptr<MetaRunContext> &run_context,
                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);

    int32_t CheckLink(const std::shared_ptr<MetaRunContext> &run_context,
                      const std::vector<std::shared_ptr<FlowMsg> > &input_msgs);

    int32_t AllocateCache(const std::shared_ptr<MetaRunContext> &run_context,
                          const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t DeallocateCache(const std::shared_ptr<MetaRunContext> &run_context,
                            const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t RemoveCacheIndex(const std::shared_ptr<MetaRunContext> &run_context,
                             const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t CopyCache(const std::shared_ptr<MetaRunContext> &run_context,
                      const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t PullCache(const std::shared_ptr<MetaRunContext> &run_context,
                      const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t TransferCache(const std::shared_ptr<MetaRunContext> &run_context,
                     const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t GetCache(const std::shared_ptr<MetaRunContext> &run_context,
                     const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t SyncKvData(const std::shared_ptr<MetaRunContext> &run_context,
                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t SwitchRole(const std::shared_ptr<MetaRunContext> &run_context,
                       const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);

private:
    enum class FlowFuncType : size_t {
      kUpdateLink = 0,
      kUnlink = 1,
      kAllocate = 2,
      kDeallocate = 3,
      kGetTensorSummary = 4,
      kGetTensor = 5,
      kCopy = 6,
      kRemoveIndex = 7,
      kPull = 8,
      kSwitchRole = 9,
      kTransfer = 10,
      kInitialize = 11,
      kCheckLink = 12,
      kMax = 13,
    };
    static int32_t SetOutput(const std::shared_ptr<MetaRunContext> &run_context, uint32_t output_idx, FsmStatus status);
    int32_t SetOutput(const std::shared_ptr<MetaRunContext> &run_context, FlowFuncType flow_func_type, FsmStatus status);
    int32_t ParseFlowFuncAttrs(const std::shared_ptr<MetaParams> &params);
    int32_t InitClusterManager(const std::shared_ptr<MetaParams> &params, bool is_prompt);
    void FinalizeStatistic();
    void ResetStatistic(const std::string &role);
    static void SetEnvs(const std::shared_ptr<MetaParams> &params);

    std::string role_;
    int32_t device_id_ = 0;
    void *statistic_timer_handle_ = nullptr;
    ClusterManager cluster_manager_;
    std::vector<uint32_t> flow_func_out_indices_;
};
}  // namespace FlowFunc

#endif  // BUILT_IN_LLM_FUNC_LLM_SERVICE_FLOW_FUNC_
