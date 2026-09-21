/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_BUILT_IN_FSM_SEND_STATE_H
#define UDF_BUILT_IN_FSM_SEND_STATE_H

#include "fsm/base_state.h"
#include "llm_common/llm_common.h"
#include "flow_func/flow_msg.h"

namespace FlowFunc {
class SendState : public BaseState {
public:
    SendState() noexcept = default;
    ~SendState() override = default;
    FsmStatus Preprocess(LlmCommEntity &entity) override;
    FsmStatus Process(LlmCommEntity &entity) override;
    FsmStatus Postprocess(LlmCommEntity &entity) override;
    SendState(const SendState &) = delete;
    SendState(const SendState &&) = delete;
    SendState &operator=(const SendState &) = delete;
    SendState &operator=(const SendState &&) = delete;

private:
    static FsmStatus CheckKvCacheManagerReq(const std::vector<KvTensor> &kv_tensors, const SyncKvReqInfo *req_info);
    static FsmStatus CheckNotBlocksReq(const SyncKvReqInfo *req_info);
    static FsmStatus GenerateSyncKvMetaInfo(LlmCommEntity &entity);
    static FsmStatus SendSyncKvMetaAsync(LlmCommEntity &entity);
    static FsmStatus SendKvCacheAsync(LlmCommEntity &entity);
    static FsmStatus TestSendRequestsAsync(LlmCommEntity &entity);
    static FsmStatus CheckSyncKvReqInfo(const SyncKvReqInfo *req_info,
                                        const std::vector<KvTensor> &kv_tensors,
                                        LlmCommEntity &entity);
    static FsmStatus QueryPromptKvCache(const SyncKvReqInfo *req_info, std::vector<KvTensor> &kv_tensors);
    static FsmStatus QueryKvCacheInCacheManager(const SyncKvReqInfo *req_info, std::vector<KvTensor> &kv_tensors);
    static FsmStatus QueryBlocksKvCache(const SyncKvReqInfo *req_info, std::vector<KvTensor> &kv_tensors);
    static FsmStatus QueryKvTensorByCacheId(const SyncKvReqInfo &req_info, std::vector<KvTensor> &kv_tensors);
    static bool GetCacheKey(const SyncKvReqInfo &req_info, std::pair<uint64_t, uint64_t> &cache_key, bool &is_prefix);
    static void ReleaseKvCacheForPrompt(const LlmCommEntity &entity);
    static FsmStatus FilterByTensorIndices(const SyncKvReqInfo &req_info, std::vector<KvTensor> &kv_tensors);
};
}  // namespace FlowFunc
#endif  // UDF_BUILT_IN_FSM_SEND_STATE_H