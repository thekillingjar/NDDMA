/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_BUILT_IN_FSM_RECEIVE_TRANSFER_REQ_STATE_H
#define UDF_BUILT_IN_FSM_RECEIVE_TRANSFER_REQ_STATE_H

#include "fsm/base_state.h"
#include "llm_common/llm_common.h"

namespace FlowFunc {
class ReceiveTransferReqState : public BaseState {
public:
    ReceiveTransferReqState() noexcept = default;
    ~ReceiveTransferReqState() override = default;

    static FsmStatus TestReq(LlmCommEntity &entity);
    static FsmStatus CheckAndSendMeta(LlmCommEntity &entity);
    static FsmStatus TestSendMeta(LlmCommEntity &entity);
    FsmStatus Preprocess(LlmCommEntity &entity) override;
    FsmStatus Process(LlmCommEntity &entity) override;
    FsmStatus Postprocess(LlmCommEntity &entity) override;
    static FsmStatus ValidateAndSetPushAddrs(const TransferToRemoteReq *req_info, LlmCommEntity &entity);
    ReceiveTransferReqState(const ReceiveTransferReqState &) = delete;
    ReceiveTransferReqState(const ReceiveTransferReqState &&) = delete;
    ReceiveTransferReqState &operator=(const ReceiveTransferReqState &) = delete;
    ReceiveTransferReqState &operator=(const ReceiveTransferReqState &&) = delete;
};
}  // namespace FlowFunc
#endif  // UDF_BUILT_IN_FSM_RECEIVE_TRANSFER_REQ_STATE_H